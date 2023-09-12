conda activate moe-adaptors
export CUDA_VISIBLE_DEVICES=3

# python main.py -plm_name roberta-base -dataset cola -graph_index 2 -output_dir output/glue_roberta_pos/cola 2>&1 | tee glue_roberta_cola_pos.out
# python main.py -plm_name roberta-base -dataset mrpc -graph_index 2 -output_dir output/glue_roberta_pos/mrpc 2>&1 | tee glue_roberta_mrpc_pos.out
# python main.py -plm_name roberta-base -dataset rte  -graph_index 2 -output_dir output/glue_roberta_pos/rte 2>&1 | tee glue_roberta_rte_pos.out
# python main.py -plm_name roberta-base -dataset stsb -graph_index 2 -output_dir output/glue_roberta_pos/stsb 2>&1 | tee glue_roberta_stsb_pos.out
python main.py -per_device_train_batch_size 16 -gradient_accumulation_steps 1 -plm_name roberta-base -dataset sst2 -graph_index 2 -output_dir output/glue_roberta_pos/sst2 2>&1 | tee glue_roberta_sst2_pos.out
# python main.py -plm_name roberta-base -dataset qnli -graph_index 2 -output_dir output/glue_roberta_pos/qnli 2>&1 | tee glue_roberta_qnli_pos.out
# python main.py -plm_name roberta-base -dataset mnli -graph_index 2 -output_dir output/glue_roberta_pos/mnli 2>&1 | tee glue_roberta_mnli_pos.out
# python main.py -plm_name roberta-base -dataset qqp  -graph_index 2 -output_dir output/glue_roberta_pos/qqp 2>&1 | tee glue_roberta_qqp_pos.out
