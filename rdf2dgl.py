from datasets import load_dataset, DatasetDict
import numpy as np
import evaluate
import torch
import dgl
import os
import time
import argparse

from data.graphs import load_rdf_graphs
from data.semantic_dataset import SemanticDataset
from data.parse import create_syntax_graphs, create_chain_graphs

from tqdm import tqdm


glue_task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    # "wnli": ("sentence1", "sentence2"),
}

glue_task_to_dirname = {
    "cola": "CoLA",
    "mnli": "MNLI",
    "mrpc": "MRPC",
    "qnli": "QNLI",
    "qqp": "QQP",
    "rte": "RTE",
    "sst2": "SST-2",
    "stsb": "STS-B",
    # "wnli": None,
}

def main(args):

    path = os.path.join(args.rdf_dir, glue_task_to_dirname[args.name])
    has_secondary_split = False

    semantic_graphs, relation2id, num_sem_relations = load_rdf_graphs(path, 'dm', has_secondary_split)

    for split in semantic_graphs.keys():
        if split == 'train':
            save_split_name = 'train'
        elif split == 'dev':
            save_split_name = 'validation'
        elif split == 'test':
            save_split_name = 'test'
        else: # For dev2
            continue

        split_dir = os.path.join(args.save_dir, name, save_split_name, 'dm')
        os.makedirs(split_dir, exist_ok=True)

        save_count = 0
        for i in range(0, len(semantic_graphs[split]), 1000):
            end = i + 1000
            if end > len(semantic_graphs[split]):
                end = len(semantic_graphs[split])
            torch.save(semantic_graphs[split][i:end], os.path.join(split_dir, f'{save_count}.pt'))
            save_count += 1
            print(f"Saved {i}-{end} out of {len(semantic_graphs[split])}.")

        print(f"Loaded and saved {args.name}-{save_split_name}({split}) graphs!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-rdf_dir', default='./resources/glue_graphs') 
    parser.add_argument('-save_dir', default='./resources/dgl_graphs') 
    args = parser.parse_args()

    for name in glue_task_to_keys.keys():
        args.name = name
        main(args)