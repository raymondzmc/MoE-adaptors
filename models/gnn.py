import torch
import torch.nn as nn
import numpy as np
from dgl.nn.pytorch import RelGraphConv

import pdb, time

class RGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_relations, num_bases, num_hidden_layers=1, dropout=0.1, activation=nn.ReLU):
        super().__init__()
        self.down_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList()
        self.activation = activation()
        self.up_proj = nn.Linear(hidden_dim, output_dim)

        # Initialize GNN layers
        for i in range(num_hidden_layers + 1):
            if i == 0:
                dim1, dim2 = input_dim, hidden_dim
            elif i == num_hidden_layers:
                dim1, dim2 = hidden_dim, output_dim
            else:
                dim1, dim2 = hidden_dim, hidden_dim

            self.layers.append(
                RelGraphConv(
                    hidden_dim,
                    hidden_dim,
                    num_relations,
                    regularizer=None,
                    num_bases=None,
                    # regularizer="basis",
                    # num_bases=num_bases,
                    activation=activation(),
                    self_loop=True,
                    dropout=dropout,
                )
            )

    def forward(self, x, graphs, add_residual=False, residual=None):
        x = self.down_proj(x)
        sent_a_masks = graphs['sent_a_masks']
        graphs_a = graphs['graphs_a']
        gdata_a = graphs['gdata_a']

        node_embs_a, node_emb_mask_a = pool_node_embeddings(x, sent_a_masks, gdata_a, graphs_a.batch_num_nodes())
        node_embs_a = self.propagate_graph(graphs_a, node_embs_a, node_emb_mask_a)
        
        no_graph_nodes = torch.ones_like(sent_a_masks)
        concatenated_rep = torch.cat((x, node_embs_a), dim=1)
        # graph_indices = []
        # results = []
        
        
        
            # results.append(concatenated_rep[i][x_indices])
        # concatenated_rep
            # for idx in indices:
            #     x[i][sent_a_masks[i]][idx[0]] = node_embs_a[i][node_emb_mask_a[i]][idx[1]]
        
        # node_embs_a[i][graph_indices]

        if graphs['graphs_b'] != None:
            sent_b_masks = graphs['sent_b_masks']
            graphs_b = graphs['graphs_b']
            gdata_b = graphs['gdata_b']
            node_embs_b, node_emb_mask_b = pool_node_embeddings(x, sent_b_masks, gdata_b, graphs_b.batch_num_nodes())
            node_embs_b = self.propagate_graph(graphs_b, node_embs_b, node_emb_mask_b)
            concatenated_rep = torch.cat((concatenated_rep, node_embs_b), dim=1)

        select_indices = []
        for i in range(len(x)):
            x_indices = list(range(x.shape[1]))
            indice_pairs_a = gdata_a['wpidx2graphid'][i].nonzero().tolist()
            graph_indices_a = [x[1] for x in indice_pairs_a]
            wp_indices_a = [x[0] for x in indice_pairs_a]
            sent_a_masks[0].nonzero().squeeze().tolist()
            sent_a_indices = sent_a_masks[i].nonzero().squeeze().tolist()
            sent_a_indices = sent_a_indices if isinstance(sent_a_indices, list) else [sent_a_indices]
            wp_indices_a = [sent_a_indices[x] for x in wp_indices_a]
            for wp_idx, graph_idx in zip(wp_indices_a, graph_indices_a):
                x_indices[wp_idx] = x.shape[1] + graph_idx

            if graphs['graphs_b'] != None:
                indice_pairs_b = gdata_b['wpidx2graphid'][i].nonzero().tolist()
                graph_indices_b = [x[1] for x in indice_pairs_b]
                wp_indices_b = [x[0] for x in indice_pairs_b]
                sent_b_indices = sent_b_masks[i].nonzero().squeeze().tolist()
                sent_b_indices = sent_b_indices if isinstance(sent_b_indices, list) else [sent_b_indices]
                wp_indices_b = [sent_b_indices[x] for x in wp_indices_b]
                for wp_idx, graph_idx in zip(wp_indices_b, graph_indices_b):
                    x_indices[wp_idx] = x.shape[1] + node_embs_a.shape[1] + graph_idx
            
        
            # results.append(torch.stack((node_embs_a[i][graph_indices], x[i][no_graph_nodes[i]])))
            select_indices.append(x_indices)
        
        out = torch.stack([concatenated_rep[i][idx] for i, idx in enumerate(select_indices)])
        out = self.up_proj(out)

        return out
    
    def propagate_graph(self, graph, node_embeddings, node_embeddings_mask):
        """
        Parameters:
            node_embs: (bsz, max_num_nodes, emb_dim)
            node_embeddings_mask: (bsz, max_num_nodes)

        Returns:
            node_embs: (bsz, max_num_nodes, emb_dim)
        """
        node_embeddings = self.flatten_node_embeddings(node_embeddings, node_embeddings_mask)
        node_embeddings = self.activation(node_embeddings)
        
        graph = graph.to(node_embeddings.device)
        for layer in self.layers:
            types = torch.zeros_like(graph.edata['type'])
            types[graph.edata['type'] < 0] = 1
            graph.edata['type'] = types
            node_embeddings = layer(graph,
            node_embeddings,
            graph.edata['type'] if 'type' in graph.edata else h.new_empty(0),
            graph.edata['norm'] if 'norm' in graph.edata else h.new_empty(0),
        )

        return self.unflatten_node_embeddings(node_embeddings, node_embeddings_mask)

    def interact_graphs(self, graph_a, graph_b, node_embs_a, node_embs_b, node_emb_mask_a, node_emb_mask_b):
        """
        Parameters:
            node_embs_{a,b}: (bsz, n_nodes_{a,b}, graph_dim)
            node_emb_mask_{a,b}: (bsz, n_nodes_{a,b})
        """
        orig_node_embs_a, orig_node_embs_b = node_embs_a, node_embs_b

        # attn: (bsz, n_nodes_a, n_nodes_b)
        attn = self.attn_biaffine(node_embs_a, node_embs_b)

        normalized_attn_a = masked_softmax(attn, node_emb_mask_a.unsqueeze(2), dim=1)  # (bsz, n_nodes_a, n_nodes_b)
        attended_a = normalized_attn_a.transpose(1, 2).bmm(node_embs_a)  # (bsz, n_nodes_b, graph_dim)
        new_node_embs_b = torch.cat([node_embs_b, attended_a, node_embs_b - attended_a, node_embs_b * attended_a], dim=-1)  # (bsz, n_nodes_b, graph_dim * 4)
        new_node_embs_b = self.activation(self.attn_proj(new_node_embs_b))  # (bsz, n_nodes_b, graph_dim)

        normalized_attn_b = masked_softmax(attn, node_emb_mask_b.unsqueeze(1), dim=2)  # (bsz, n_nodes_a, n_nodes_b)
        attended_b = normalized_attn_b.bmm(node_embs_b)  # (bsz, n_nodes_a, graph_dim)
        new_node_embs_a = torch.cat([node_embs_a, attended_b, node_embs_a - attended_b, node_embs_a * attended_b], dim=-1)  # (bsz, n_nodes_a, graph_dim * 4)
        new_node_embs_a = self.activation(self.attn_proj(new_node_embs_a))  # (bsz, n_nodes_b, graph_dim)

        node_embs_a = self.flatten_node_embeddings(new_node_embs_a, node_emb_mask_a)
        node_embs_b = self.flatten_node_embeddings(new_node_embs_b, node_emb_mask_b)

        node_embs_a = self.unflatten_node_embeddings(node_embs_a, node_emb_mask_a)
        node_embs_b = self.unflatten_node_embeddings(node_embs_b, node_emb_mask_b)

        # If the other graph is empty, we don't do any attention at all and use the original embedding
        node_embs_a = torch.where(node_emb_mask_b.any(1, keepdim=True).unsqueeze(-1), node_embs_a, orig_node_embs_a)
        node_embs_b = torch.where(node_emb_mask_a.any(1, keepdim=True).unsqueeze(-1), node_embs_b, orig_node_embs_b)

        return node_embs_a, node_embs_b

    def pool_graph(self, node_embs, node_emb_mask):
        """
        Parameters:
            node_embs: (bsz, n_nodes, graph_dim)
            node_emb_mask: (bsz, n_nodes)

        Returns:
            (bsz, graph_dim (*2))
        """
        node_emb_mask = node_emb_mask.unsqueeze(-1)
        output = masked_max(node_embs, node_emb_mask, 1)
        output = torch.where(node_emb_mask.any(1), output, torch.zeros_like(output))
        return output

    def flatten_node_embeddings(self, node_embeddings, node_embeddings_mask):
        return node_embeddings[node_embeddings_mask]

    def unflatten_node_embeddings(self, node_embeddings, node_embeddings_mask):
        output_node_embeddings = node_embeddings.new_zeros(
            node_embeddings_mask.shape[0], node_embeddings_mask.shape[1], node_embeddings.shape[-1]
        )
        output_node_embeddings[node_embeddings_mask] = node_embeddings
        return output_node_embeddings


def first_true_idx(tensor, dim, cumsum=None):
    # Takes a bool tensor and returns the idx of the first element that is True in a given dimension
    # Undefined return value iff all elements in the given dimension are False
    # The implementation is adapted from https://discuss.pytorch.org/t/first-nonzero-index/24769/9
    cumsum = cumsum if cumsum is not None else tensor.cumsum(dim)
    return ((cumsum == 1) & tensor).max(dim)[1]


def last_true_idx(tensor, dim, cumsum=None):
    # Takes a bool tensor and returns the idx of the last element that is True in a given dimension
    # Returns 0 if all elements in the given dimension are False
    cumsum = cumsum if cumsum is not None else tensor.cumsum(dim)
    return first_true_idx(cumsum == cumsum.max(dim)[0].unsqueeze(dim), dim)


def masked_mean(vector, mask, dim, keepdim=False) -> torch.Tensor:
    """
    To calculate mean along certain dimensions on masked values
    # Parameters
    vector : `torch.Tensor`
        The vector to calculate mean.
    mask : `torch.BoolTensor`
        The mask of the vector. It must be broadcastable with vector.
    dim : `int`
        The dimension to calculate mean
    keepdim : `bool`
        Whether to keep dimension
    # Returns
    `torch.Tensor`
        A `torch.Tensor` of including the mean values.
    """
    replaced_vector = vector.masked_fill(~mask, 0.0)

    value_sum = torch.sum(replaced_vector, dim=dim, keepdim=keepdim)
    value_count = torch.sum(mask, dim=dim, keepdim=keepdim)
    return value_sum / value_count.float().clamp(min=1e-13)

def pool_node_embeddings(last_layers, masks, gdata, batch_num_nodes):
        """
        Convert wordpiece embeddings into word (i.e. node) embeddings using the alignment in
        wpidx2graphid = gdata['wpidx2graphid']

        Parameters:
            g_data: dictinoary with values having shape:
                (bsz, ...)
            masks: (bsz, max_sent_pair_len)
            last_layers: (bsz, max_sent_pair_len, emb_dim)

        Returns:
            node_embs: (bsz, max_num_nodes, emb_dim)
            node_embeddings_mask: (bsz, max_num_nodes)
        """
        wpidx2graphid = gdata['wpidx2graphid']  # (bsz, max_sent_len, max_n_nodes)
        device = last_layers.device
        bsz, max_sent_len, max_n_nodes = wpidx2graphid.shape
        emb_dim = last_layers.shape[-1]
        try:
            assert max(batch_num_nodes) == wpidx2graphid.shape[-1]
        except:
            pdb.set_trace()

        # the following logic happens to work if the graph is empty, in which case its sentence_end is guaranteed to be 1 (exclusive)
        masks_cumsum = masks.cumsum(1)
        sentence_starts = first_true_idx(masks, 1, masks_cumsum)
        sentence_ends = last_true_idx(masks, 1, masks_cumsum) + 1  # exclusive
        max_sentence_len = (sentence_ends - sentence_starts).max()

        # we're using a for loop here since only doing rolling across the batch dimension shouldn't be very expensive
        # that said, can we do it without a loop?
        rolled_last_layers = torch.stack([last_layer.roll(-sentence_start.item(), dims=0) for last_layer, sentence_start in zip(last_layers, sentence_starts)])
        segmented_last_layers = rolled_last_layers[:, :max_sentence_len, :]  # (bsz, max_sent_len, emb_dim)
        
        try:
            assert segmented_last_layers.shape[:2] == wpidx2graphid.shape[:2]
        except:
            pdb.set_trace()

        # (bsz, max_sent_len, max_n_nodes, emb_dim)
        expanded_wpidx2graphid = wpidx2graphid.unsqueeze(-1).expand(-1, -1, -1, emb_dim)
        expanded_segmented_last_layers = segmented_last_layers.unsqueeze(2).expand(-1, -1, max_n_nodes, -1)

        # (bsz, max_n_nodes, emb_dim)
        node_embeddings = masked_mean(expanded_segmented_last_layers, expanded_wpidx2graphid, 1)

        node_embeddings = torch.where(expanded_wpidx2graphid.any(1), node_embeddings, torch.tensor(0., device=device))  # some nodes don't have corresponding wordpieces
        node_embeddings_mask = torch.arange(max(batch_num_nodes), device=device).expand(bsz, -1) < torch.tensor(batch_num_nodes, dtype=torch.long, device=device).unsqueeze(1)

        return node_embeddings, node_embeddings_mask