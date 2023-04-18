conda activate moe-adaptors
export CUDA_VISIBLE_DEVICES=1

# python main.py -plm_name roberta-base -adator_type moe -expert_type gnn -dataset cola -gate_type softmax -output_dir output/glue/roberta/moe_gnn_softmax_cola/ 2>&1 | tee glue_roberta_moe_gnn_softmax_cola.out
python main.py -plm_name roberta-base -adator_type moe -expert_type gnn -dataset mrpc -gate_type softmax -output_dir output/glue/roberta/moe_gnn_softmax_mrpc/ 2>&1 | tee glue_roberta_moe_gnn_softmax_mrpc.out
python main.py -plm_name roberta-base -adator_type moe -expert_type gnn -dataset rte -gate_type softmax -output_dir output/glue/roberta/moe_gnn_softmax_rte/ 2>&1 | tee glue_roberta_moe_gnn_softmax_rte.out
python main.py -plm_name roberta-base -adator_type moe -expert_type gnn -dataset stsb -gate_type softmax -output_dir output/glue/roberta/moe_gnn_softmax_stsb/ 2>&1 | tee glue_roberta_moe_gnn_softmax_stsb.out
# python main.py -plm_name roberta-base -adator_type moe -expert_type gnn -dataset qnli -gate_type softmax -output_dir output/glue/roberta/moe_gnn_softmax_qnli/ 2>&1 | tee glue_roberta_moe_gnn_softmax_qnli.out
# python main.py -plm_name roberta-base -adator_type moe -expert_type gnn -dataset sst2 -gate_type softmax -output_dir output/glue/roberta/moe_gnn_softmax_sst2/ 2>&1 | tee glue_roberta_moe_gnn_softmax_sst2.out
