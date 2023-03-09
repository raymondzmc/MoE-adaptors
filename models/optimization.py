import math
import torch
import transformers


def get_optimizer(args, model):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    if args.optim == 'adamw_hf':
        optimizer = transformers.AdamW(
            optimizer_grouped_parameters,
            lr = args.learning_rate,
            betas = (args.adam_beta1, args.adam_beta2),
            eps = args.adam_epsilon,
            weight_decay = args.weight_decay,
        )
    
    return optimizer


def get_lr_scheduler(args, optimizer, train_dataset_len):

    if args.max_steps > 0:
        max_train_steps = args.max_steps
    else:
        true_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
        if args.dataloader_drop_last:
            num_update_steps_per_epoch = math.floor(train_dataset_len / true_batch_size)
        else:
            num_update_steps_per_epoch = math.ceil(train_dataset_len / true_batch_size)

        max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = transformers.get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=max_train_steps,
    )
    
    return lr_scheduler