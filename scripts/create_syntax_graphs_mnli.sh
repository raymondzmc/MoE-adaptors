#!/bin/bash
#SBATCH --gres=gpu:v100l:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=3  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=64000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=1-0:00
#SBATCH --output=create_syntax_graph_mnli.out

module load gcc/9.3.0 arrow python/3.9 scipy-stack
source /home/liraymo6/virtualenvs/moe-adaptors/bin/activate
python data/parse.py -graph_types syntax -dataset mnli -resource_dir ./resources/dgl_graphs
