import numpy as np
import evaluate
from transformers import EvalPrediction
import pdb


glue_tasks = ["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]

def get_glue_metrics(task_name):

    is_regression = task_name == "stsb"
    metric = evaluate.load("glue", task_name)

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if isinstance(p.label_ids, list):
            p.label_ids = p.label_ids[0]
        if task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}
    
    return compute_metrics


def get_compute_metric(name):
    if name in glue_tasks:
        return get_glue_metrics(name)


