import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction
from models.gnn import RGCN
import time

import math
import pdb

"""
Implementation based on https://github.com/lucidrains/mixture-of-experts
"""

def default(val, default_val):
    default_val = default_val() if isfunction(default_val) else default_val
    return val if val is not None else default_val

eps = 1e-9

class SoftmaxGating(nn.Module):
    def __init__(self, input_dim, num_gates, gate_type='softmax'):
        super().__init__()
        self.num_gates = num_gates
        self.w_gating = nn.Linear(input_dim, num_gates)
        self.gate_type = gate_type

    def forward(self, x):
        raw_gates = self.w_gating(x.mean(1))

        if self.gate_type == 'softmax':
            softmax_gates = raw_gates.softmax(dim=-1)
        elif self.gate_type == 'gumbel':
            softmax_gates = F.gumbel_softmax(raw_gates, tau=1, hard=True, dim=-1)
        return softmax_gates

class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation=nn.GELU, adapter_scalar='4', dropout=0):
        super().__init__()

        self.adapter_layer_norm_before = nn.LayerNorm(input_dim)
        self.down_proj = nn.Linear(input_dim, hidden_dim)
        self.non_linear_func = activation()
        self.up_proj = nn.Linear(hidden_dim, output_dim)

        self.scale = float(adapter_scalar)
        self.dropout = dropout

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up_proj.weight)
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.bias)
        
        

    def forward(self, x, add_residual=True, residual=None, **kwargs):
        residual = x if residual is None else residual
        x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = F.dropout(down, p=self.dropout)
        up = self.up_proj(down)

        up = up * self.scale

        # if self.adapter_layernorm_option == 'out':
        #     up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output

class MoE_Adaptor(nn.Module):
    def __init__(self,
        input_dim,
        num_experts=4,
        hidden_dim=512,
        activation=nn.ReLU,
        adapter_scalar='1',
        dropout=0,
        loss_coef=1e-2,
        expert_type='mlp',
        num_relations=1,
        num_bases=80,
        gate_type='softmax',
    ):
        super().__init__()
        
        self.down_proj = nn.Linear(input_dim, hidden_dim)
        self.up_proj = nn.Linear(hidden_dim, input_dim)
        self.num_experts = num_experts
        self.num_gnn_experts = 3

        self.gate = SoftmaxGating(input_dim, num_gates=num_experts, gate_type=gate_type)
        
        output_dim = input_dim
        
        if expert_type == 'mlp':
            self.experts = nn.ModuleList([
                Expert(
                    hidden_dim,
                    hidden_dim=hidden_dim,
                    output_dim=hidden_dim,
                    activation=activation,
                    adapter_scalar=adapter_scalar,
                    dropout=dropout,
                ) for _ in range(num_experts)
            ])
        elif expert_type == 'gnn':
            self.experts = nn.ModuleList([
                RGCN(
                    hidden_dim,
                    hidden_dim,
                    hidden_dim,
                    num_relations=num_relations,
                    num_bases=num_bases,
                    num_hidden_layers=1,
                    dropout=0.1,
                    activation=activation,
                ) if i < self.num_gnn_experts else 
                Expert(
                    hidden_dim,
                    hidden_dim=hidden_dim,
                    output_dim=hidden_dim,
                    activation=activation,
                    adapter_scalar=adapter_scalar,
                    dropout=dropout,
                ) for i in range(num_experts)
            ])
        else:
            raise NotImplementedError(f"Expert type \"{expert_type}\" not implemented!")

    def forward(self, x, add_residual=False, residual=None, graphs=None):
        
        gates = self.gate(x)
        x = self.down_proj(x)
        # t0 = time.time()
        
        # t1 = time.time()
        # print(f"Gate: {t1-t0}")
        experts_output = []
        for i, expert in enumerate(self.experts):
            # t0  = time.time()
            g = graphs[i] if graphs != None and i < self.num_gnn_experts else None
            out = expert(x, add_residual=add_residual, residual=residual, graphs=g)
            # t1 = time.time()
            experts_output.append(out)
            # print(f"Expert {i}: {t1-t0}")
        # experts_output = torch.stack([
        #     expert(x, add_residual=add_residual, residual=residual, graphs=graphs) for expert in self.experts
        # ]).permute(1, 0, 2, 3) # batch, expert, sequence_len, hidden
        # return experts_output.squeeze(1)
        # pdb.set_trace()
        output = (gates.unsqueeze(-1).unsqueeze(-1) * torch.stack(experts_output).permute(1, 0, 2, 3)).sum(1)
        # output = gates.unsqueeze(-1).unsqueeze(-1) * experts_output.sum(1)
        output = self.up_proj(output)
        # # bz, n_experts, seq_len, h = output.shape
        # # output = output.permute(0, 2, 1, 3).reshape(bz, seq_len, n_experts * h)
        return output, gates
