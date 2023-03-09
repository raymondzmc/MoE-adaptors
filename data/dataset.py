import evaluate
from datasets import load_dataset

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

        

def get_dataset(name, tokenizer):

    if name in glue_task_to_keys.keys():
        dataset = load_dataset("glue", name)
        is_regression = (name == "stsb")
        if not is_regression:
            label_list = dataset["train"].features["label"].names
            label_to_id = {v: i for i, v in enumerate(label_list)}
            num_labels = len(label_list)
        else:
            label_to_id = None
            num_labels = 1
        
        sentence1_key, sentence2_key = glue_task_to_keys[name]

        def preprocess_function(examples):
            if sentence2_key is None:
                texts = (list(map(lambda x: x.strip().split(' '), examples[sentence1_key])),)
            else:
                sentence1 = list(map(lambda x: " ".join(x.split()).strip().split(' '), examples[sentence1_key]))
                sentence2 = list(map(lambda x: " ".join(x.split()).strip().split(' '), examples[sentence2_key]))
                texts = (sentence1, sentence2)

            result = tokenizer(*texts, padding=True, max_length=128, truncation=True, is_split_into_words=True)
            if "label" in examples:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]

                # if label_to_id is not None:
                #     # Map labels to IDs (not necessary for GLUE tasks)
                #     result["labels"] = [label_to_id[l] for l in examples["label"]]
                # else:
                    
            return result

        processed_datasets = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
            desc="Running tokenizer on dataset",
        )

        train_dataset = processed_datasets["train"]
        eval_dataset = processed_datasets["validation_matched" if name == "mnli" else "validation"]

        compute_metric = evaluate.load('glue', name)
        return train_dataset, eval_dataset, compute_metric, num_labels