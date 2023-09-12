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
    def __init__(self, input_dim, num_gates, gate_type='softmax', tau=1):
        super().__init__()
        self.num_gates = num_gates
        self.w_gating = nn.Linear(input_dim, num_gates)

        # if gate_type == 'gumbel':
        #     self.
        self.gate_type = gate_type
        self.tau = tau

    def forward(self, x, tau=None):
        raw_gates = self.w_gating(x.mean(1))

        if self.gate_type == 'softmax':
            softmax_gates = raw_gates.softmax(dim=-1)
        elif self.gate_type == 'gumbel':
            if tau == None:
                tau = self.tau
            softmax_gates = F.gumbel_softmax(raw_gates, tau=tau, hard=False, dim=-1)
        return softmax_gates

class HardConcreteGating(nn.Module):
    def __init__(self, input_dim, num_gates, gate_type='softmax', tau=1):
        super().__init__()
        self.num_gates = num_gates
        self.w_gating = nn.Linear(input_dim, num_gates)
        
        self.gate_type = gate_type
        self.tau = tau

    def forward(self, x, tau=None):
        raw_gates = self.w_gating(x)

        if self.gate_type == 'softmax':
            softmax_gates = raw_gates.softmax(dim=-1)
        elif self.gate_type == 'gumbel':
            if tau == None:
                tau = self.tau
            softmax_gates = F.gumbel_softmax(raw_gates, tau=tau, hard=False, dim=-1)
        return softmax_gates

class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation=nn.GELU, adapter_scalar='learnable_scalar', dropout=0):
        super().__init__()

        self.adapter_layer_norm_before = nn.LayerNorm(input_dim)
        self.down_proj = nn.Linear(input_dim, hidden_dim)
        self.non_linear_func = activation()
        self.up_proj = nn.Linear(hidden_dim, output_dim)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
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
        tau=1.0,
        graph_index=None,
    ):
        super().__init__()
        
        # self.down_proj = nn.Linear(input_dim, hidden_dim)
        # self.up_proj = nn.Linear(hidden_dim, input_dim)
        self.num_experts = num_experts
        self.num_gnn_experts = 3
        self.gate_logits = nn.Parameter(torch.zeros(self.num_experts))
        self.use_gates = True
        # self.gate = SoftmaxGating(input_dim, num_gates=num_experts, gate_type=gate_type, tau=tau)
        self.graph_index = graph_index
        output_dim = input_dim
        if expert_type == 'mlp':
            self.experts = nn.ModuleList([
                Expert(
                    input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim,
                    activation=activation,
                    adapter_scalar=adapter_scalar,
                    dropout=dropout,
                ) for _ in range(num_experts)
            ])
        elif expert_type == 'gnn':
            if graph_index == None:
                self.experts = nn.ModuleList([
                    RGCN(
                        input_dim,
                        hidden_dim,
                        output_dim,
                        num_relations=num_relations,
                        num_bases=num_bases,
                        num_hidden_layers=1,
                        dropout=0.1,
                        activation=activation,
                    ) if i < self.num_gnn_experts else 
                    Expert(
                        input_dim,
                        hidden_dim=hidden_dim,
                        output_dim=output_dim,
                        activation=activation,
                        adapter_scalar=adapter_scalar,
                        dropout=dropout,
                    ) for i in range(num_experts)
                ])
            else:
                self.use_gates = False
                if self.graph_index < self.num_gnn_experts:
                    self.experts = RGCN(
                        input_dim,
                        hidden_dim,
                        output_dim,
                        num_relations=num_relations,
                        num_bases=num_bases,
                        num_hidden_layers=1,
                        dropout=0.1,
                        activation=activation,
                    )
                else:
                    self.experts = Expert(
                        input_dim,
                        hidden_dim=hidden_dim,
                        output_dim=output_dim,
                        activation=activation,
                        adapter_scalar=adapter_scalar,
                        dropout=dropout,
                    )
        else:
            raise NotImplementedError(f"Expert type \"{expert_type}\" not implemented!")

        
        
    
    def remove_experts(self):
        if isinstance(self.experts, nn.ModuleList):
            self.graph_index = torch.argmax(self.gate_logits).item()
            setattr(self, 'experts', self.experts[self.graph_index])
            self.use_gates = False

    def forward(self, x, add_residual=False, residual=None, graphs=None, tau=None, one_hot_gate=True):
        
        # x = self.down_proj(x)
        # t0 = time.time()
        
        # t1 = time.time()
        # print(f"Gate: {t1-t0}")
        experts_output = []
        if self.graph_index != None:
            g = graphs[self.graph_index] if graphs != None and self.graph_index < self.num_gnn_experts else None
            experts_output = self.experts(x, add_residual=add_residual, residual=residual, graphs=g)
        else:
            for i, expert in enumerate(self.experts):
                g = graphs[i] if graphs != None and i < self.num_gnn_experts else None
                out = expert(x, add_residual=add_residual, residual=residual, graphs=g)
                experts_output.append(out)

        # F.gumbel_softmax(experts_output)
        # experts_output = self.gate(experts_output.permute(1, 0, 2, 3))
        # pdb.set_trace()
        if self.use_gates:
            if self.training:
                batch_size = x.shape[0]
                expanded_gate_logits = self.gate_logits.unsqueeze(1).expand(-1, batch_size)
                gates = F.gumbel_softmax(expanded_gate_logits, tau=tau, hard=False, dim=0)[:, :, None, None]
            else:
                # Take the expert with the single highest probability
                if one_hot_gate:
                    arg_max = torch.argmax(self.gate_logits)
                    one_hot = torch.zeros_like(self.gate_logits)
                    one_hot[arg_max] = 1
                    gates = one_hot[:, None, None, None]
                else:
                    gates = self.gate_logits.softmax(dim=0)[:, None, None, None]
                    # gates = self.gate.softmax(dim=-1).unsqueeze(0).unsqueeze(0).unsqueeze(0)
                    # print(self.gate, gates[0, 0, 0].tolist())
                    # gates = F.gumbel_softmax(self.gate, tau=tau, hard=True, dim=-1).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            output = (torch.stack(experts_output) * gates).sum(0)
        else:
            output = experts_output
            gates = self.gate_logits.softmax(dim=0)[:, None, None, None]
        
        # print(tau)
        # gates = self.gate(experts_output tau=tau, hard=False, dim=-1)
        # experts_output = torch.stack([
        #     expert(x, add_residual=add_residual, residual=residual, graphs=graphs) for expert in self.experts
        # ]).permute(1, 0, 2, 3) # batch, expert, sequence_len, hidden
        # output = (gates.unsqueeze(-1) * experts_output).sum(1)
        # output = gates.unsqueeze(-1).unsqueeze(-1) * experts_output.sum(1)
        # output = self.up_proj(output)
        # # bz, n_experts, seq_len, h = output.shape
        # # output = output.permute(0, 2, 1, 3).reshape(bz, seq_len, n_experts * h)
        return output, gates.detach()
