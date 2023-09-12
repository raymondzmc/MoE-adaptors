conda activate moe-adaptors
export CUDA_VISIBLE_DEVICES=2

# python main.py -plm_name roberta-base -dataset cola -graph_index 1 -output_dir output/glue_roberta_syn/cola 2>&1 | tee glue_roberta_cola_syn.out
# python main.py -plm_name roberta-base -dataset mrpc -graph_index 1 -output_dir output/glue_roberta_syn/mrpc 2>&1 | tee glue_roberta_mrpc_syn.out
# python main.py -plm_name roberta-base -dataset rte  -graph_index 1 -output_dir output/glue_roberta_syn/rte 2>&1 | tee glue_roberta_rte_syn.out
# python main.py -plm_name roberta-base -dataset stsb -graph_index 1 -output_dir output/glue_roberta_syn/stsb 2>&1 | tee glue_roberta_stsb_syn.out
python main.py -per_device_train_batch_size 16 -gradient_accumulation_steps 1 -plm_name roberta-base -dataset sst2 -graph_index 1 -output_dir output/glue_roberta_syn/sst2 2>&1 | tee glue_roberta_sst2_syn.out
# python main.py -plm_name roberta-base -dataset qnli -graph_index 1 -output_dir output/glue_roberta_syn/qnli 2>&1 | tee glue_roberta_qnli_syn.out
# python main.py -plm_name roberta-base -dataset mnli -graph_index 1 -output_dir output/glue_roberta_syn/mnli 2>&1 | tee glue_roberta_mnli_syn.out
# python main.py -plm_name roberta-base -dataset qqp  -graph_index 1 -output_dir output/glue_roberta_syn/qqp 2>&1 | tee glue_roberta_qqp_syn.out
