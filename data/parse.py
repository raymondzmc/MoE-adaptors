import os
import stanza
import torch
import numpy as np
from dgl import DGLGraph
import argparse
from datasets import load_dataset, DatasetDict

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
    "wnli": ("sentence1", "sentence2"),
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
    "wnli": None,
}


import pdb

def create_chain_graphs(dataset, sentence1_key, sentence2_key=None):
    graphs = {}

    nlp = stanza.Pipeline(lang='en', processors='tokenize', tokenize_pretokenized=True)
    for split in dataset.keys():
        subset = dataset[split]
        graphs[split] = []
        
        for example in tqdm(subset):

            for key in [sentence1_key, sentence2_key]:
                if key == None:
                    continue
                
                
                metadata = []
                doc = nlp(example[key])
                
                sentence = doc.sentences[0] # Assume single sentence input
                num_nodes = len(sentence.words)
                edge_list = []
                for i, word in enumerate(sentence.words):
                    metadata.append({
                        'anchor': [word.start_char, word.end_char],
                    })
                    if i < (num_nodes - 1):
                        edge_list.append((i, i + 1, -1))
                        edge_list.append((i + 1, i, 1))

                if num_nodes == 0:
                    g = DGLGraph()
                    g.gdata = {'metadata': metadata}
                    graphs[split].append(g)
                    continue
                
                edge_list = sorted(edge_list, key=lambda x: (x[1], x[0], x[2]))
                edge_list = np.array(edge_list, dtype=int)

                if len(edge_list):
                    edge_src, edge_dst, edge_type = edge_list.transpose()
                else:
                    edge_src, edge_dst, edge_type = np.array([]), np.array([]), np.array([])

                # normalize by dst degree
                _, inverse_index, count = np.unique((edge_dst, edge_type), axis=1, return_inverse=True, return_counts=True)
                degrees = count[inverse_index]
                edge_norm = np.ones(len(edge_dst), dtype=np.float32) / degrees.astype(np.float32)

                node_ids = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
                edge_type = torch.from_numpy(edge_type)
                edge_norm = torch.from_numpy(edge_norm).unsqueeze(1)
                g = DGLGraph()
                g.add_nodes(num_nodes)
                g.add_edges(edge_src, edge_dst)
                g.ndata.update({'id': node_ids})
                g.edata.update({'type': edge_type, 'norm': edge_norm})
                g.gdata = {'metadata': metadata}
                graphs[split].append(g)
    return graphs, {'next': 0, 'prev': 1}, 2


def create_syntax_graphs(dataset, sentence1_key, sentence2_key=None):
    graphs = {}

    nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse', tokenize_pretokenized=True)
    relation2id = {}
    relations_counter = 1 # Since we are using negatives
    max_rel = 100

    for split in dataset.keys():
        subset = dataset[split]
        graphs[split] = []
        
        for i, example in enumerate(tqdm(subset)):

            for key in [sentence1_key, sentence2_key]:
                if key == None:
                    continue
                
                
                metadata = []
                
                doc = nlp(example[key])
                
                sentence = doc.sentences[0] # Assume single sentence input
                num_nodes = len(sentence.words)
                edge_list = []
                for word in sentence.words:
                    metadata.append({
                        'anchor': [word.start_char, word.end_char],
                        'upos': word.upos,
                        'xpos': word.xpos,
                    })

                    rel = word.deprel
                    
                    # No edge to add
                    if rel == 'root':
                        continue

                    # Add new relations to dictionary
                    if not rel in relation2id:
                        relation2id[rel] = relations_counter
                        relations_counter += 1
                    
                    index = word.id - 1
                    head = word.head - 1
                    edge_list.append((head, index, relation2id[rel]))
                    edge_list.append((index, head, -relation2id[rel]))

                if num_nodes == 0:
                    g = DGLGraph()
                    g.gdata = {'metadata': metadata}
                    graphs[split].append(g)
                    continue
                
                edge_list = sorted(edge_list, key=lambda x: (x[1], x[0], x[2]))
                edge_list = np.array(edge_list, dtype=int)

                if len(edge_list):
                    edge_src, edge_dst, edge_type = edge_list.transpose()
                else:
                    edge_src, edge_dst, edge_type = np.array([]), np.array([]), np.array([])

                # normalize by dst degree
                _, inverse_index, count = np.unique((edge_dst, edge_type), axis=1, return_inverse=True, return_counts=True)
                degrees = count[inverse_index]
                edge_norm = np.ones(len(edge_dst), dtype=np.float32) / degrees.astype(np.float32)

                node_ids = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
                edge_type = torch.from_numpy(edge_type)
                edge_norm = torch.from_numpy(edge_norm).unsqueeze(1)
                g = DGLGraph()
                g.add_nodes(num_nodes)
                g.add_edges(edge_src, edge_dst)
                g.ndata.update({'id': node_ids})
                g.edata.update({'type': edge_type, 'norm': edge_norm})
                g.gdata = {'metadata': metadata}
                graphs[split].append(g)
    
    return graphs, relation2id, relations_counter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-graph_types', default='all', choices=['all', 'syntax', 'chain']) 
    parser.add_argument('-dataset', default='all', choices=list(glue_task_to_keys.keys()) + ['all']) 
    parser.add_argument('-resource_dir', default='../resources/dgl_graphs') 
    args = parser.parse_args()

    os.makedirs(args.resource_dir, exist_ok=True)
    batch_size = 1000

    for task_name in glue_task_to_keys.keys():
        task_dir = os.path.join(args.resource_dir, task_name)

        if args.dataset != 'all' and task_name != args.dataset:
            continue

        dataset = load_dataset("glue", task_name)
        sentence1_key, sentence2_key = glue_task_to_keys[task_name]

        if args.graph_types in ['all', 'chain']:
            graphs, relation2id, num_relations = create_chain_graphs(dataset, sentence1_key, sentence2_key)

            for split in graphs.keys():
                save_path = os.path.join(task_dir, split, 'chain')
                os.makedirs(save_path, exist_ok=True)
                save_count = 0
                
                for i in range(0, len(graphs[split]), batch_size):
                    end = i + batch_size
                    if end > len(graphs[split]):
                        end = len(graphs[split])
                    torch.save(graphs[split][i:end], os.path.join(save_path, f'{save_count}.pt'))
                    save_count += 1
                    print(f"Saved {i}-{end} out of {len(graphs[split])}.")
                print("Loaded and saved Chain graphs!")

        if args.graph_types in ['all', 'syntax']:
            graphs, relation2id, num_relations = create_syntax_graphs(dataset, sentence1_key, sentence2_key)

            for split in graphs.keys():
                save_path = os.path.join(task_dir, split, 'syntax')
                os.makedirs(save_path, exist_ok=True)
                save_count = 0
                for i in range(0, len(graphs[split]), batch_size):
                    end = i + batch_size
                    if end > len(graphs[split]):
                        end = len(graphs[split])
                    torch.save(graphs[split][i:end], os.path.join(save_path, f'{save_count}.pt'))
                    save_count += 1
                    print(f"Saved {i}-{end} out of {len(graphs[split])}.")
                print("Loaded and saved Syntax graphs!")
        
        
