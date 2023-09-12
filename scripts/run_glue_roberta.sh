conda activate moe-adaptors
export CUDA_VISIBLE_DEVICES=1

python main.py -plm_name roberta-base -dataset cola -output_dir output/glue_roberta/cola 2>&1 | tee glue_roberta_cola.out
python main.py -plm_name roberta-base -dataset mrpc -output_dir output/glue_roberta/mrpc 2>&1 | tee glue_roberta_mrpc.out
python main.py -plm_name roberta-base -dataset rte  -output_dir output/glue_roberta/rte 2>&1 | tee glue_roberta_rte.out
python main.py -plm_name roberta-base -dataset stsb -output_dir output/glue_roberta/stsb 2>&1 | tee glue_roberta_stsb.out
python main.py -plm_name roberta-base -dataset sst2 -output_dir output/glue_roberta/sst2 2>&1 | tee glue_roberta_sst2.out
python main.py -plm_name roberta-base -dataset qnli -output_dir output/glue_roberta/qnli 2>&1 | tee glue_roberta_qnli.out
python main.py -plm_name roberta-base -dataset mnli -output_dir output/glue_roberta/mnli 2>&1 | tee glue_roberta_mnli.out
python main.py -plm_name roberta-base -dataset qqp  -output_dir output/glue_roberta/qqp 2>&1 | tee glue_roberta_qqp.out
