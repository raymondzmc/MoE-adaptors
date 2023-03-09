import os
import argparse

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, get_scheduler

from data.collators import default_data_collator, DataCollatorWithPadding
from data.dataset import get_dataset
from trainer import Trainer
from utils.trainer import get_trainer_arguments

from models.optimization import get_optimizer, get_lr_scheduler
import pdb


def main(args):

    tokenizer = AutoTokenizer.from_pretrained(args.plm_name)
    
    train_dataset, eval_dataset, compute_metrics, num_labels = get_dataset(args.dataset, tokenizer)
    
    config = AutoConfig.from_pretrained(args.plm_name, num_labels=num_labels, finetuning_task=args.dataset)
    model = AutoModelForSequenceClassification.from_pretrained(args.plm_name, config=config)
    
    if args.pad_to_max_length:
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
    

if __name__ == '__main__':
    CWD = '.'

    parser = argparse.ArgumentParser()
    parser.add_argument('-device', default='0', type=str) # Used string for cpu
    parser.add_argument('-seed', default=42, type=int)
    

    # Model specific arguments
    parser.add_argument('-plm_name', default='bert-base-uncased', type=str, help='Name or path of pre-trained model for AutoModel')

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
    parser.add_argument('-num_train_epochs', default=3, type=int, help='Total number of training epochs to perform.')
    parser.add_argument('-max_steps', default=-1, type=int, help='If set to a positive number, the total number of training steps to perform. Overrides num_train_epochs.')
    parser.add_argument('-save_strategy', default='epoch', type=str, choices=['no', 'epoch', 'steps'], help='The checkpoint save strategy to adopt during training.')
    parser.add_argument('-save_steps', default=500, type=int, help='Number of updates steps before two checkpoint saves if save_strategy=\'steps\'')
    parser.add_argument('-resume_from_checkpoint', default=False, help='The path to a folder with a valid checkpoint for your model (Not directly used by Trainer).')

    args = parser.parse_args()

    main(args)