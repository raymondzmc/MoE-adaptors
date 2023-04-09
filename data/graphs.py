
import os, sys, pickle
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import numpy as np
from dgl import DGLGraph
from rdflib import Literal
from tqdm import tqdm

import pdb


def load_rdf_graphs(path, formalism, has_secondary_split):
    return get_graphs(path, formalism, has_secondary_split)


def get_graphs(
    data_dir,
    formalism,
    has_secondary_split=False,
    verbose=False,
):  
    # Load picked RDF graphs (https://rdflib.readthedocs.io/en/stable/intro_to_graphs.html)
    train_rdf_graphs = pickle.load(open(os.path.join(data_dir, f'train.{formalism}.rdf'), 'rb'))
    if verbose: print("Loaded training graph")
    dev_rdf_graphs = pickle.load(open(os.path.join(data_dir, f'dev.{formalism}.rdf'), 'rb'))
    if verbose: print("Loaded dev graph")
    test_rdf_graphs = pickle.load(open(os.path.join(data_dir, f'test.{formalism}.rdf'), 'rb'))
    if verbose: print("Loaded test graph")
    if has_secondary_split:
        dev2_rdf_graphs = pickle.load(open(os.path.join(data_dir, f'dev2.{formalism}.rdf'), 'rb'))
        test2_rdf_graphs = pickle.load(open(os.path.join(data_dir, f'test2.{formalism}.rdf'), 'rb'))

    # Load metadata files
    train_metadata = pickle.load(open(os.path.join(data_dir, f'train.{formalism}.metadata'), 'rb'))
    dev_metadata = pickle.load(open(os.path.join(data_dir, f'dev.{formalism}.metadata'), 'rb'))
    test_metadata = pickle.load(open(os.path.join(data_dir, f'test.{formalism}.metadata'), 'rb'))
    if has_secondary_split:
        dev2_metadata = pickle.load(open(os.path.join(data_dir, f'dev2.{formalism}.metadata'), 'rb'))
        test2_metadata = pickle.load(open(os.path.join(data_dir, f'test2.{formalism}.metadata'), 'rb'))

    # Check for number of examples
    assert len(train_rdf_graphs) == len(train_metadata)
    assert len(dev_rdf_graphs) == len(dev_metadata)
    assert len(test_rdf_graphs) == len(test_metadata)
    if has_secondary_split:
        assert len(dev2_rdf_graphs) == len(dev2_metadata)
        assert len(test2_rdf_graphs) == len(test2_metadata)
    
    # Create list of split names and nested list of metadata/graphs
    all_split_names = ['train', 'dev', 'test']
    if has_secondary_split:
        all_split_names.extend(['dev2', 'test2'])
    all_rdf_graphs = [train_rdf_graphs, dev_rdf_graphs, test_rdf_graphs]
    if has_secondary_split:
        all_rdf_graphs.extend([dev2_rdf_graphs, test2_rdf_graphs])
    all_metadata = [train_metadata, dev_metadata, test_metadata]
    if has_secondary_split:
        all_metadata.extend([dev2_metadata, test2_metadata])

    return all_rdf_to_dgl(all_split_names, all_rdf_graphs, all_metadata)


def all_rdf_to_dgl(all_split_names, all_rdf_graphs, all_metadata, bidirectional=True):
    """
    Outer loop that iterate over each split (e.g. train, dev, test), 
    and convert RDF graphs (and corresponding metadata) to DGL graphs
    """
    assert len(all_split_names) == len(all_rdf_graphs) == len(all_metadata)
    
    # Get the set of all relations and their index mapping
    relations, total_graphs = relations_in(_flatten(all_rdf_graphs))
    relation2id = {rel: i for i, rel in enumerate(sorted(relations))}
    print(f'Relations count: {len(relations)}')

    graphs = {}
    
    pbar = tqdm(total=total_graphs)

    # Iterate over all splits and convert RDF to DGL graphs
    for split, rdf_graphs, metadata in zip(all_split_names, all_rdf_graphs, all_metadata):
        assert len(rdf_graphs) == len(metadata)
        graphs[split] = []
        for rdf_graph, mdata in zip(rdf_graphs, metadata):
            graph = rdf2dgl(
                rdf_graph, mdata, relation2id, bidirectional=bidirectional
            )
            graphs[split].append(graph)
            pbar.update(1)
    pbar.close()
    return graphs, relation2id, len(relations) * (2 if bidirectional else 1)

def _flatten(l):
    """
    Flatten a nested list
    """
    return [e for subl in l for e in subl]

class RDFReader(object):
    """
    Object for reading the RDF graph
    """
    __graph = None
    __freq = {}

    def __init__(self, graph):
        self.__graph = graph
        self.__freq = Counter(self.__graph.predicates())

    def triples(self, relation=None):
        """
        Generator for all triples in the format of <subject, predicate, object>
        """
        for s, p, o in self.__graph.triples((None, relation, None)):
            yield s, p, o

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.__graph.destroy("store")
        self.__graph.close(True)

    def subjectSet(self):
        """
        Return the set of subjects
        """
        return set(self.__graph.subjects())

    def objectSet(self):
        """
        Return the set of objects
        """
        return set(self.__graph.objects())

    def relationList(self):
        """
        Returns a list of relations, ordered descending by frequency
        :return:    
        """
        res = list(set(self.__graph.predicates()))
        res.sort(key=lambda rel: - self.freq(rel))
        return res

    def __len__(self):
        return len(self.__graph)

    def freq(self, rel):
        """
        Return the frequence for relation "rel"
        """
        if rel not in self.__freq:
            return 0
        return self.__freq[rel]

def relations_in(rdf_graphs):
    """
    Get the set of all relations from a list of RDF graphs
    """
    all_relations = set()
    for rdf_graph in tqdm(rdf_graphs):
        with RDFReader(rdf_graph) as reader:
            all_relations |= set(reader.relationList())
    return all_relations, len(rdf_graphs)

def rdf2dgl(rdf_graph, metadata, relation2id, bidirectional=True):
    """
    Convert a RDF graph and its corresponding metadata to a DGL graph
    """
    assert set(relation2id.values()) == set(range(len(relation2id)))

    with RDFReader(rdf_graph) as reader:

        # List/set of all relations, subjects, objects
        relations = reader.relationList()
        subjects = reader.subjectSet()
        objects = reader.objectSet()

        # Get all nodes (objects and subjects) and relations
        nodes = sorted(list(subjects.union(objects)))
        assert [int(node) for node in nodes] == list(range(len(nodes)))  # to make sure the metadata-node alignment is correct
        num_node = len(nodes)
        assert num_node == len(metadata)
        num_rel = len(relations)
        num_rel = 2 * num_rel # * 2 for bi-directionality

        # Return an empty graph
        if num_node == 0:
            g = DGLGraph()
            g.gdata = {'metadata': metadata}
            return g

        assert num_node < np.iinfo(np.int32).max

        edge_list = []
        
        # Iterate over all triples to create the list of edges
        for i, (s, p, o) in enumerate(reader.triples()):
            assert int(s) < num_node and int(o) < num_node
            rel = relation2id[p]
            edge_list.append((int(s), int(o), rel))
            if bidirectional:
                # Two types of edges (parent vs child) for bidirectional graphs
                edge_list.append((int(o), int(s), -rel))

        # sort indices by destination
        edge_list = sorted(edge_list, key=lambda x: (x[1], x[0], x[2]))
        edge_list = np.array(edge_list, dtype=int)

    # List of source node, destination node, and edge types in the same format as:
    # https://docs.dgl.ai/guide/graph-graphs-nodes-edges.html
    edge_src, edge_dst, edge_type = edge_list.transpose()

    # normalize by dst degree
    _, inverse_index, count = np.unique((edge_dst, edge_type), axis=1, return_inverse=True, return_counts=True)
    degrees = count[inverse_index]
    edge_norm = np.ones(len(edge_dst), dtype=np.float32) / degrees.astype(np.float32)

    node_ids = torch.arange(0, num_node, dtype=torch.long).view(-1, 1)
    edge_type = torch.from_numpy(edge_type)
    edge_norm = torch.from_numpy(edge_norm).unsqueeze(1)

    # See documentation: https://docs.dgl.ai/api/python/dgl.DGLGraph.html
    g = DGLGraph()
    g.add_nodes(num_node)
    g.add_edges(edge_src, edge_dst)
    g.ndata.update({'id': node_ids})
    g.edata.update({'type': edge_type, 'norm': edge_norm})

    g.gdata = {'metadata': metadata}  # we add this field in DGLGraph

    return g