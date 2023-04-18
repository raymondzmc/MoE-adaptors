conda activate moe-adaptors
export CUDA_VISIBLE_DEVICES=0

python main.py -plm_name roberta-base -per_device_train_batch_size 4 -gradient_accumulation_steps 16 -freeze_plm -pooling_method cls -adator_type moe -expert_type gnn -dataset mnli -gate_type gumbel -output_dir output/glue/roberta/moe_gnn_gumbel_mnli_cls/ 2>&1 | tee glue_roberta_moe_gnn_gumbel_mnli_cls.out
