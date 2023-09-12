conda activate moe-adaptors
export CUDA_VISIBLE_DEVICES=0

# python main.py -plm_name roberta-base -dataset cola -graph_index 0 -output_dir output/glue_roberta_sem/cola 2>&1 | tee glue_roberta_cola_sem.out
# python main.py -plm_name roberta-base -dataset mrpc -graph_index 0 -output_dir output/glue_roberta_sem/mrpc 2>&1 | tee glue_roberta_mrpc_sem.out
# python main.py -plm_name roberta-base -dataset rte  -graph_index 0 -output_dir output/glue_roberta_sem/rte 2>&1 | tee glue_roberta_rte_sem.out
# python main.py -plm_name roberta-base -dataset stsb -graph_index 0 -output_dir output/glue_roberta_sem/stsb 2>&1 | tee glue_roberta_stsb_sem.out

# python main.py -per_device_train_batch_size 16 -gradient_accumulation_steps 1 \
#       -plm_name roberta-base -dataset sst2 -graph_index custom \
#       -output_dir output/glue_roberta_custom/sst2 2>&1 | tee glue_roberta_sst2_custom.out

python main.py -per_device_train_batch_size 16 -gradient_accumulation_steps 1 \
     -plm_name roberta-base -dataset qnli -graph_index custom \
     -output_dir output/glue_roberta_custom/qnli 2>&1 | tee glue_roberta_qnli_custom.out
# python main.py -plm_name roberta-base -dataset mnli -graph_index 0 -output_dir output/glue_roberta_sem/mnli 2>&1 | tee glue_roberta_mnli_sem.out
# python main.py -plm_name roberta-base -dataset qqp  -graph_index 0 -output_dir output/glue_roberta_sem/qqp 2>&1 | tee glue_roberta_qqp_sem.out
