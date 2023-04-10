#!/bin/bash
#SBATCH --gres=gpu:v100l:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=192000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=2-0:00
#SBATCH --output=glue_moe_gnn_softmax.out

module load gcc/9.3.0 arrow python/3.9 scipy-stack
source /home/liraymo6/virtualenvs/moe-adaptors/bin/activate

python main.py -plm_name roberta-base -adator_type moe -expert_type gnn -dataset cola -output_dir output/glue/roberta/moe_gnn_softmax_cola/
# python main.py -plm_name roberta-base -adator_type moe -expert_type gnn -dataset mnli -output_dir output/glue/roberta/moe_gnn_softmax_mnli/
python main.py -plm_name roberta-base -adator_type moe -expert_type gnn -dataset mrpc -output_dir output/glue/roberta/moe_gnn_softmax_mrpc/
python main.py -plm_name roberta-base -adator_type moe -expert_type gnn -dataset sst2 -output_dir output/glue/roberta/moe_gnn_softmax_sst2/
# python main.py -plm_name roberta-base -adator_type moe -expert_type gnn -dataset stsb -output_dir output/glue/roberta/moe_gnn_softmax_stsb/
# python main.py -plm_name roberta-base -adator_type moe -expert_type gnn -dataset qqp -output_dir output/glue/roberta/moe_gnn_softmax_qqp/
# python main.py -plm_name roberta-base -adator_type moe -expert_type gnn -dataset qnli -output_dir output/glue/roberta/moe_gnn_softmax_qnli/
# python main.py -plm_name roberta-base -adator_type moe -expert_type gnn -dataset rte -output_dir output/glue/roberta/moe_gnn_softmax_rte/