conda activate moe-adaptors
export CUDA_VISIBLE_DEVICES=2

python main.py -plm_name roberta-base -adator_type moe -expert_type gnn -dataset cola -gate_type gumbel -output_dir output/glue/roberta/moe_gnn_gumbel_cola/ 2>&1 | tee glue_roberta_moe_gnn_gumbel_cola.out
python main.py -plm_name roberta-base -adator_type moe -expert_type gnn -dataset mrpc -gate_type gumbel -output_dir output/glue/roberta/moe_gnn_gumbel_mrpc/ 2>&1 | tee glue_roberta_moe_gnn_gumbel_mrpc.out
python main.py -plm_name roberta-base -adator_type moe -expert_type gnn -dataset sst2 -gate_type gumbel -output_dir output/glue/roberta/moe_gnn_gumbel_sst2/ 2>&1 | tee glue_roberta_moe_gnn_gumbel_sst2.out
python main.py -plm_name roberta-base -adator_type moe -expert_type gnn -dataset rte -gate_type gumbel -output_dir output/glue/roberta/moe_gnn_gumbel_rte/ 2>&1 | tee glue_roberta_moe_gnn_gumbel_rte.out
python main.py -plm_name roberta-base -adator_type moe -expert_type gnn -dataset stsb -gate_type gumbel -output_dir output/glue/roberta/moe_gnn_gumbel_stsb/ 2>&1 | tee glue_roberta_moe_gnn_gumbel_stsb.out
python main.py -plm_name roberta-base -adator_type moe -expert_type gnn -dataset qnli -gate_type gumbel -output_dir output/glue/roberta/moe_gnn_gumbel_qnli/ 2>&1 | tee glue_roberta_moe_gnn_gumbel_qnli.out