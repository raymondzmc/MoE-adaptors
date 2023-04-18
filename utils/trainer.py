from transformers import TrainingArguments


def get_trainer_arguments(args):
    """
    Return the trainer argument based on args, for the complete list, refer to:
    https://huggingface.co/docs/transformers/v4.26.1/en/main_classes/trainer#transformers.TrainingArguments
    """
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        logging_dir=args.output_dir,
        overwrite_output_dir=True,
        do_train=args.do_train,
        do_eval=True,
        do_predict=True,
        evaluation_strategy=args.evaluation_strategy,
        prediction_loss_only=False,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr_scheduler_type='linear',
        eval_accumulation_steps=1,
        eval_delay=0,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        save_strategy=args.save_strategy,
        save_steps=1,
        seed=args.seed,
        data_seed=args.seed,
        dataloader_drop_last=False,
        run_name='',
        disable_tqdm=False,
        remove_unused_columns=True,
        save_total_limit=args.save_total_limit
    )

    return training_args