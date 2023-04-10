import os
import argparse

import transformers
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, get_scheduler

from data.collators import default_data_collator, DataCollatorWithPadding
from data.dataset import get_dataset
from evaluation.metrics import get_compute_metric
from trainer import Trainer
from utils.trainer import get_trainer_arguments

from models.optimization import get_optimizer, get_lr_scheduler
from models.modeling_roberta import RobertaForSequenceClassification
from models.petl.petl_enc_model import PETLEncModel
from models.petl.options import TuneArguments

import seaborn as sns
import matplotlib.pyplot as plt

import pdb


glue_output_modes = {
    "cola": "classification",
    "mnli": "classification",
    "mnli-mm": "classification",
    "mrpc": "classification",
    "sst2": "classification",
    "stsb": "regression",
    "qqp": "classification",
    "qnli": "classification",
    "rte": "classification",
    "wnli": "classification",
}

def main(args):
    # tokenizer = AutoTokenizer.from_pretrained(args.plm_name)
    tokenizer = AutoTokenizer.from_pretrained(args.plm_name, add_prefix_space=True) # RoBERTa tokenizer
    
    load_graphs = args.expert_type == 'gnn'
    train_dataset, eval_dataset, compute_metrics, num_labels, num_relations = get_dataset(args.dataset, tokenizer, load_graphs=load_graphs)
    compute_metrics = get_compute_metric(args.dataset)

    config = AutoConfig.from_pretrained(args.plm_name, num_labels=num_labels, finetuning_task=args.dataset)
    config.attn_mode="none"
    config.attn_option="parallel"
    config.attn_composition="add"
    config.attn_bn=200  # attn bottleneck dim (not used for parallel Scaled PA)
    config.ffn_mode="adapter"
    config.ffn_option="parallel"
    config.ffn_adapter_layernorm_option="none"
    config.ffn_adapter_init_option="lora"
    config.ffn_adapter_scalar="4"
    config.ffn_bn=256 # ffn bottleneck dim
    config.adaptor_type = args.adator_type
    config.expert_type = args.expert_type
    config.gate_type = args.gate_type

    # GNN arguments
    config.num_relations = num_relations
    config.num_bases = 80


    
    model = RobertaForSequenceClassification.from_pretrained(args.plm_name, config=config)
    
    tune_args = TuneArguments(
        attn_mode="none",
        attn_option="parallel",
        attn_composition="add",
        attn_bn=200,  # attn bottleneck dim (not used for parallel Scaled PA)
        ffn_mode="adapter",
        ffn_option="parallel",
        ffn_adapter_layernorm_option="none",
        ffn_adapter_init_option="lora",
        ffn_adapter_scalar="4",
        ffn_bn=512, # ffn bottleneck dim
    )
    
    model = PETLEncModel(config, tune_args, model)
    if load_graphs:
        data_collator = lambda batch: train_dataset.collate_fn(
                batch, tokenizer.pad_token_id, tokenizer.pad_token_type_id, tokenizer.padding_side,
                glue_output_modes[args.dataset],
            )
    elif args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorWithPadding(tokenizer)
    


    optimizer = get_optimizer(args, model)
    lr_scheduler = get_lr_scheduler(args, optimizer, len(train_dataset))
    trainer_args = get_trainer_arguments(args)
    
    trainer = Trainer(
        model=model,
        args=trainer_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset, 
        tokenizer=tokenizer, 
        compute_metrics=compute_metrics,
        optimizers=(optimizer, lr_scheduler)
    )
    trainer.train()
    
    evaluate = False
    if evaluate:
        output, gates = trainer.evaluate(eval_dataset, output_gates=True)

        # Plot gates:
        gates = gates.mean(0).cpu().numpy()
        sns.heatmap(gates, cmap='viridis', annot=True)
        
        # Customize the plot, if desired
        plt.title('Gate Values for GLUE')
        plt.xlabel('Semantic (DM), Syntax, Positional (Chain)')
        plt.ylabel('Layers')

        # Save the heatmap
        plt.savefig('gates_heatmap.png')

    # else:
    #     checkpoint_dir = sorted(os.listdir(args.output_dir), lambda x: int(x.split('-')[-1]))[0]:


    

if __name__ == '__main__':
    CWD = '.'

    parser = argparse.ArgumentParser()
    parser.add_argument('-device', default='0', type=str) # Used string for cpu
    parser.add_argument('-seed', default=42, type=int)
    

    # Model specific arguments
    parser.add_argument('-plm_name', default='bert-base-uncased', type=str, help='Name or path of pre-trained model for AutoModel')
    parser.add_argument('-adator_type', default='mlp', type=str, choices=['mlp', 'moe'])
    parser.add_argument('-expert_type', default='mlp', type=str, choices=['mlp', 'gnn'])
    parser.add_argument('-gate_type', default='softmax', type=str, choices=['softmax', 'gumbel'])
    parser.add_argument('-load_graphs', action='store_true')

    # Dataset/Dataloader Arguments
    parser.add_argument('-dataset', default=None, type=str, required=True, help='Name or path of dataset')
    parser.add_argument('-per_device_train_batch_size', default=8, type=int)
    parser.add_argument('-per_device_eval_batch_size', default=8, type=int)
    parser.add_argument('-dataloader_num_workers', default=0, type=int, help='Number of workers in DataLoader, default to zero for Iterable Dataset')
    parser.add_argument('-dataloader_drop_last', action='store_true', help='Whether to drop the last incomplete batch.')
    parser.add_argument('-pad_to_max_length', action='store_true')


    # Optimizer/LR Scheduler Arguments
    parser.add_argument('-optim', default='adamw_hf', type=str, choices=['adamw_hf', 'adamw_torch', 'adamw_apex_fused', 'adamw_anyprecision', 'adafactor'])
    parser.add_argument('-learning_rate', default=5e-5, type=float)
    parser.add_argument('-weight_decay', default=0, type=float)
    parser.add_argument('-adam_beta1', default=0.9, type=float)
    parser.add_argument('-adam_beta2', default=0.999, type=float)
    parser.add_argument('-adam_epsilon', default=1e-8, type=float)
    parser.add_argument('-lr_scheduler_type', default='linear', type=str, choices=['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup'])
    parser.add_argument('-warmup_ratio', default=0, type=float)
    parser.add_argument('-warmup_steps', default=0, type=int)
    

    # Trainer arguments
    parser.add_argument('-output_dir', default=None, type=str, required=True, help='The output directory where the model predictions and checkpoints will be written.')
    parser.add_argument('-do_train', action='store_true', help='Whether to run training or not. (Not directly used by Trainer)')
    parser.add_argument('-do_eval', action='store_true', help='Whether to run evaluation on the validation set  or not. (Not directly used by Trainer)')
    parser.add_argument('-do_predict', action='store_true', help='Whether to run predictions on the test set or not. (Not directly used by Trainer)')
    parser.add_argument('-evaluation_strategy', default='epoch', choices=['no', 'steps', 'epoch'], help='The evaluation strategy to adopt during training.')
    parser.add_argument('-gradient_accumulation_steps', default=8, type=int, help='Number of updates steps to accumulate the gradients for, before performing a backward/update pass.')
    parser.add_argument('-num_train_epochs', default=10, type=int, help='Total number of training epochs to perform.')
    parser.add_argument('-max_steps', default=-1, type=int, help='If set to a positive number, the total number of training steps to perform. Overrides num_train_epochs.')
    parser.add_argument('-save_strategy', default='epoch', type=str, choices=['no', 'epoch', 'steps'], help='The checkpoint save strategy to adopt during training.')
    parser.add_argument('-save_steps', default=500, type=int, help='Number of updates steps before two checkpoint saves if save_strategy=\'steps\'')
    parser.add_argument('-resume_from_checkpoint', default=None, help='The path to a folder with a valid checkpoint for your model (Not directly used by Trainer).')

    args = parser.parse_args()

    main(args)