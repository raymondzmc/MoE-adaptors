import dgl
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
import pdb


def GraphDataset(Dataset):
    def __init__(self, input_ids, attention_mask, token_type_ids=None, graphs=None):
        assert len(input_ids) == len(attention_mask) == len(token_type_ids) == len(labels) == len(sent_a_masks[0]) == len(graphs_a[0])
        # assert num_graphs == len(sent_a_masks) == len(graphs_a) 
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.labels = labels


        self.sent_a_masks = sent_a_masks
        self.sent_b_masks = sent_b_masks
        self.graphs_a = graphs_a
        self.graphs_b = graphs_b
        self.num_graphs = num_graphs

    def __getitem__(self, index):
        return [
            self.input_ids[index],
            self.attention_mask[index],
            self.token_type_ids[index],
            [self.sent_a_masks[i][index] for i in range(self.num_graphs)],
            [self.sent_b_masks[i][index] for i in range(self.num_graphs)] if self.sent_b_masks else None,
            self.labels[index],
            [self.graphs_a[i][index] for i in range(self.num_graphs)],
            [self.graphs_b[i][index] for i in range(self.num_graphs)] if self.graphs_b else None,
        ] 
