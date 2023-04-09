from datasets import load_dataset, DatasetDict
import numpy as np
import evaluate
import torch
import dgl
import os
import time

from data.graphs import load_rdf_graphs
from data.semantic_dataset import SemanticDataset
from data.parse import create_syntax_graphs, create_chain_graphs

import pdb

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


def _spans_overlap(span_a, span_b):
    return not (span_a[1] <= span_b[0] or span_a[0] >= span_b[1])  # not no_overlap


def _assert_sorted(spans, allow_overlap=False):
    if spans is None:
        return
    for i, span in enumerate(spans):
        assert span[1] >= span[0]  # we allow empty spans
        if i > 0:
            if allow_overlap:
                assert span[0] >= spans[i - 1][1] or _spans_overlap(span, spans[i - 1])
            else:
                assert span[0] >= spans[i - 1][1]


def _determine_special_token_positions(special_token_indices):
    special_token_indices = sorted(special_token_indices)

    initial_special_tokens_start = special_token_indices[0]
    initial_special_tokens_end = -1
    for i, idx in enumerate(special_token_indices):
        if i != idx:
            initial_special_tokens_end = special_token_indices[i - 1] + 1
            break
    assert initial_special_tokens_end != -1

    final_special_tokens_start = -1
    final_special_tokens_end = special_token_indices[-1] + 1
    for i, idx in enumerate(special_token_indices[::-1]):
        if i + idx + 1 != final_special_tokens_end:
            final_special_tokens_start = special_token_indices[-i]
            break
    assert final_special_tokens_start != -1

    middle_special_tokens = [
        idx for idx in special_token_indices if not (
            initial_special_tokens_start <= idx < initial_special_tokens_end
        ) and not (
            final_special_tokens_start <= idx < final_special_tokens_end
        )
    ]
    if middle_special_tokens:
        middle_special_tokens_start = middle_special_tokens[0]
        middle_special_tokens_end = middle_special_tokens[-1] + 1
        assert middle_special_tokens == list(range(middle_special_tokens_start, middle_special_tokens_end))
    else:
        middle_special_tokens_start = middle_special_tokens_end = -1

    return initial_special_tokens_start, initial_special_tokens_end, middle_special_tokens_start, middle_special_tokens_end, final_special_tokens_start, final_special_tokens_end


def _split_offsets(wp_offsets, is_pair, special_token_indices):
    """
    [(0, 0), (0, 5), (5, 6), (0, 0), (0, 4), (0, 0)] ->
    [[(0, 5), (5, 6)], [(0, 4)]]

    This function assumes there is one and only one place of consecutive special tokens between the two sentences
    """
    (
        initial_special_tokens_start, initial_special_tokens_end, middle_special_tokens_start, middle_special_tokens_end, final_special_tokens_start, final_special_tokens_end
    ) = _determine_special_token_positions(special_token_indices)
    assert initial_special_tokens_start == 0 and final_special_tokens_end == len(wp_offsets)

    if not is_pair:
        assert middle_special_tokens_start == middle_special_tokens_end == -1
        offsets_a_span = (initial_special_tokens_end, final_special_tokens_start)
        offsets_b_span = (0, 0)
    else:
        offsets_a_span = (initial_special_tokens_end, middle_special_tokens_start)
        offsets_b_span = (middle_special_tokens_end, final_special_tokens_start)

    offsets_a = wp_offsets[offsets_a_span[0]:offsets_a_span[1]]
    offsets_b = wp_offsets[offsets_b_span[0]:offsets_b_span[1]]

    # Some tokenizers, e.g. RoBERTa, split some unicode characters in weird ways,
    # so we allow local non-sorted spans if they overlap
    _assert_sorted(offsets_a, allow_overlap=True)
    _assert_sorted(offsets_b, allow_overlap=True)

    return offsets_a, offsets_b, offsets_a_span, offsets_b_span


def _calc_wpidx2graphid(anchors, wp_offsets):
    """
    Parameters:
        anchors: List[Optional[Tuple[int, int]]]
        wp_offsets: List[Tuple[int, int]]

    Returns:
        List[List[bool]]
    """
    wpidx2graphid = [[False] * len(anchors) for _ in range(len(wp_offsets))]
    # There's probably an O(n) way to do this but the lists are usually short anyway
    for wp_idx, wp_span in enumerate(wp_offsets):
        for graph_id, node_span in enumerate(anchors):
            if node_span is not None and _spans_overlap(wp_span, node_span):
                wpidx2graphid[wp_idx][graph_id] = True

    return wpidx2graphid


def _pad_and_stack_gdata(all_gdata, pad_value=False):
    max_shapes = {}
    for gdata in all_gdata:
        for k, tensor in gdata.items():
            if k not in max_shapes:
                max_shapes[k] = list(tensor.shape)
            else:
                max_shape = max_shapes[k]
                for i, (max_, curr) in enumerate(zip(max_shape, tensor.shape)):
                    max_shape[i] = max(max_, curr)

    inner_output = {k: [] for k in max_shapes.keys()}
    for inner_idx, gdata in enumerate(all_gdata):
        for k, tensor in gdata.items():
            pad = []
            has_zero_dim = False
            for i, (max_, curr) in enumerate(zip(max_shapes[k][::-1], tensor.shape[::-1])):
                pad.extend((0, max_ - curr))
                if max_ == curr == 0:
                    has_zero_dim = True

            # There are rare cases where there exists a dim i that all gdata's dim i are 0
            # It happens when, e.g. all graphs in a batch are empty
            # F.pad will complain in that case
            if has_zero_dim:
                inner_output[k].append(tensor.new_empty(max_shapes[k]))
            else:
                inner_output[k].append(F.pad(tensor, pad, value=pad_value))

    return {k: torch.stack(tensors, 0) for k, tensors in inner_output.items()}

def batch_graphs(graphs):
    batched_graphs = dgl.batch(graphs)
    gdata = _pad_and_stack_gdata([g.gdata for g in graphs])


def process_graphs(graphs, result, tokenizer, is_pair=False, name=None):
    n_examples = len(result['input_ids'])
    sent_a_masks, sent_b_masks, graphs_a, graphs_b = [], [], [], []
    for idx in range(n_examples):
        inputs = {k: result[k][idx] for k in result if k != "offset_mapping"}
        all_special_token_ids = {tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.sep_token_id, tokenizer.cls_token_id}
        wp_offsets = result["offset_mapping"][idx]
        sent_a_mask = sent_b_mask = graph_a = graph_b = None
        special_token_indices = [i for i, input_id in enumerate(inputs["input_ids"]) if input_id in all_special_token_ids]
        wp_offsets_a, wp_offsets_b, offsets_a_span, offsets_b_span = _split_offsets(
            wp_offsets,
            is_pair=is_pair,
            special_token_indices=special_token_indices,
        )
        
        sent_a_mask = [offsets_a_span[0] <= i < offsets_a_span[1] for i in range(len(wp_offsets))]
        sent_b_mask = [offsets_b_span[0] <= i < offsets_b_span[1] for i in range(len(wp_offsets))]
        to_enumerate = []
        if is_pair:
            graph_a = graphs[idx * 2]
            graph_b = graphs[idx * 2 + 1]
            to_enumerate.extend([(graph_a, wp_offsets_a), (graph_b, wp_offsets_b)])
        else:
            graph_a = graphs[idx]
            to_enumerate.append((graph_a, wp_offsets_a))

        for graph, wp_offsets in to_enumerate:
            if graph is None: continue


            # TODO: fix this in parser
            if len(graph.gdata['metadata']) and 'anchors' in graph.gdata['metadata'][0].keys():
                anchors = [metadata.get('anchors') for metadata in graph.gdata['metadata']]
            elif len(graph.gdata['metadata']) and 'anchor' in graph.gdata['metadata'][0].keys():
                anchors = [metadata.get('anchor') for metadata in graph.gdata['metadata']]
            else:
                anchors = []
            wpidx2graphid = torch.tensor(_calc_wpidx2graphid(anchors, wp_offsets), dtype=torch.bool)  # (n_wp, n_nodes)
            if wpidx2graphid.shape[-1] != graph.batch_num_nodes().item():
                pdb.set_trace()
            
            graph.gdata['wpidx2graphid'] = wpidx2graphid # TODO: if we really want to have some fun we can make this a sparse tensor
            del graph.gdata['metadata']  # save memory
        
        sent_a_masks.append(sent_a_mask)
        graphs_a.append(graph_a)

        # if graph_b.gdata['wpidx2graphid'].shape[-2] != len([x for x in sent_b_mask if x]):
        #     pdb.set_trace()

        sent_b_masks.append(sent_b_mask)
        graphs_b.append(graph_b)
    
    return sent_a_masks, sent_b_masks, graphs_a, graphs_b


def get_dataset(name, tokenizer, load_graphs=False):
    if name in glue_task_to_keys.keys():

        raw_dataset = load_dataset("glue", name)
        is_regression = (name == "stsb")
        if not is_regression:
            label_list = raw_dataset["train"].features["label"].names
            label_to_id = {v: i for i, v in enumerate(label_list)}
            num_labels = len(label_list)
        else:
            label_to_id = None
            num_labels = 1
        
        sentence1_key, sentence2_key = glue_task_to_keys[name]


        # Load graph from directory
        if load_graphs:
            dirname = glue_task_to_dirname[name]

            assert dirname != None
                # raise NotADirectoryError(f"Directory \"{dirname}\" Not Found!")

            graph_path = f'resources/dgl_graphs/{name}'

            graphs = {}

            for split in os.listdir(graph_path):

                graphs[split] = {}
                split_path = os.path.join(graph_path, split):

                for graph_name in os.listdir(split_path):
                    graph_file_path = os.path.join(split_path, graph_name)

                    graphs[split][graph_name] = [
                        torch.load(os.path.join(graph_file_path, p)) 
                            for p in tqdm(os.listdir(graph_file_path), desc=f"Loading {graph_name} graphs in \"{split}\"")
                    ]

            # Check if there's the same number of graphs
            assert [len(graphs[split][k]) == graphs[split][graphs[split].keys()[0]] for k in graphs[split].keys()]



            
            # TO DO: only load train and dev to save time
            # load_semantic_graph = True
            # if load_semantic_graph:
            #     has_secondary_split = False

            #     # semantic_graphs, relation2id, num_sem_relations = load_rdf_graphs(path, 'dm', has_secondary_split)

            #     for split in semantic_graphs.keys():
            #         split_dir = os.path.join('resources', 'dgl_graphs', name, split, 'dm')
            #         os.makedirs(split_dir, exist_ok=True)
            #         for idx, graph in enumerate(semantic_graphs[split]):
            #             torch.save(graph, os.path.join(split_dir, f'{idx}.pt'))

                # semantic_graph_path = os.path.join(path, 'semantic_graphs.pt')
                # if os.path.exists(semantic_graph_path):
                #     syntax_graphs, relation2id, num_syn_relations = torch.load(semantic_graph_path)
                # else:
                #     has_secondary_split = (name == 'mnli')
                #     has_secondary_split = False
                #     semantic_graphs, relation2id, num_sem_relations = load_rdf_graphs(path, 'dm', has_secondary_split)
                #     pdb.set_trace()
                #     torch.save((semantic_graphs, relation2id, num_sem_relations), semantic_graph_path)
                # print("Loaded Semantic graphs!")

            # load_syntax_graph = True
            # if load_syntax_graph:
            #     syntax_graph_path = os.path.join(path, 'syntax_graphs.pt')
            #     t0 = time.time()
            #     if os.path.exists(syntax_graph_path):
            #         syntax_graphs, relation2id, num_syn_relations = torch.load(syntax_graph_path)
            #     else:
            #         syntax_graphs, relation2id, num_syn_relations = create_syntax_graphs(raw_dataset, sentence1_key, sentence2_key)
            #         # torch.save((syntax_graphs, relation2id, num_sem_relations), syntax_graph_path)
            #     t1 = time.time()
            #     print(f"Loaded Syntax graphs in {t1-t0}sec!")

            #     for split in syntax_graphs.keys():
            #         split_dir = os.path.join('resources', 'dgl_graphs', name, split, 'syntax')
            #         os.makedirs(split_dir, exist_ok=True)
            #         for idx, graph in enumerate(syntax_graphs[split]):
            #             torch.save(graph, os.path.join(split_dir, f'{idx}.pt'))


            # load_chain_graph = True
            # if load_chain_graph:
            #     chain_graph_path = os.path.join(path, 'chain_graphs.pt')
            #     t0 = time.time()
            #     if os.path.exists(chain_graph_path):
            #         chain_graphs, relation2id, num_chain_relations = torch.load(chain_graph_path)
            #     else:
            #         chain_graphs, relation2id, num_chain_relations = create_chain_graphs(raw_dataset, sentence1_key, sentence2_key)
            #         # torch.save((chain_graphs, relation2id, num_chain_relations), chain_graph_path)
            #     t1 = time.time()
            #     print(f"Loaded Chain graphs in {t1-t0}sec!")

            #     for split in chain_graphs.keys():
            #         split_dir = os.path.join('resources', 'dgl_graphs', name, split, 'chain')
            #         os.makedirs(split_dir, exist_ok=True)
            #         for idx, graph in enumerate(chain_graphs[split]):
            #             torch.save(graph, os.path.join(split_dir, f'{idx}.pt'))

        num_sem_relations = 2
        def preprocess_function(examples):
            if sentence2_key is None:
                texts = (list(map(lambda x: x.strip(), examples[sentence1_key])),)
            else:
                sentence1 = list(map(lambda x: x.strip(), examples[sentence1_key]))
                sentence2 = list(map(lambda x: x.strip(), examples[sentence2_key]))
                texts = (sentence1, sentence2)
            
            result = tokenizer(*texts, max_length=256, return_offsets_mapping=True)

            # TODO: Make token_type_ids an option argument in Dataset
            result['token_type_ids'] = [[0 for _ in range(len(x))] for x in result.encodings]
        
            if "label" in examples:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]

                # if label_to_id is not None:
                #     # Map labels to IDs (not necessary for GLUE tasks)
                #     result["labels"] = [label_to_id[l] for l in examples["label"]]
                # else:
            return result

        
        processed_datasets = raw_dataset.map(
            preprocess_function,
            batched=True,
            batch_size=None,
            # remove_columns=dataset["train"].column_names,
            desc="Running tokenizer on dataset",
        )

        if load_graphs:
            is_pair = sentence2_key is not None
            datasets = {}
            
            for split in processed_datasets.keys():
                if split not in ['train', 'validation', 'validation_matched']:
                    continue

                result = processed_datasets[split].to_dict()

                sem_sent_a_masks, sem_sent_b_masks, sem_graphs_a, sem_graphs_b = process_graphs(graphs[split]['dm'], result, tokenizer, is_pair, 'semantic')
                syn_sent_a_masks, syn_sent_b_masks, syn_graphs_a, syn_graphs_b = process_graphs(graphs[split]['syntax'],, result, tokenizer, is_pair, 'syntax')
                pos_sent_a_masks, pos_sent_b_masks, pos_graphs_a, pos_graphs_b = process_graphs(graphs[split]['chain'], result, tokenizer, is_pair, 'chain')
                datasets[split] = SemanticDataset(
                    result['input_ids'],
                    result['attention_mask'],
                    result['token_type_ids'],
                    result['labels'],
                    [sem_sent_a_masks, syn_sent_a_masks, pos_sent_a_masks],
                    [sem_sent_b_masks, syn_sent_b_masks, pos_sent_b_masks] if is_pair else None,
                    [sem_graphs_a, syn_graphs_a, pos_graphs_a],
                    [sem_graphs_b, syn_graphs_b, pos_graphs_b] if is_pair else None,
                    # [syn_sent_a_masks, pos_sent_a_masks],
                    # [syn_sent_b_masks, pos_sent_b_masks] if sentence2_key != None else None,
                    # [syn_graphs_a, pos_graphs_a],
                    # [syn_graphs_b, pos_graphs_b] if sentence2_key != None else None,
                    num_graphs=3,
                )
            processed_datasets = datasets

        train_dataset = processed_datasets["train"]
        try:
            eval_dataset = processed_datasets["validation_matched" if name == "mnli" else "validation"]
        except:
            pdb.set_trace()

        compute_metric = evaluate.load('glue', name)
        return train_dataset, eval_dataset, compute_metric, num_labels, 2