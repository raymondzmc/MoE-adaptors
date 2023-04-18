conda activate moe-adaptors
export CUDA_VISIBLE_DEVICES=0

python main.py -plm_name roberta-base -adator_type moe -expert_type gnn -dataset qnli -gate_type softmax -output_dir output/glue/roberta/moe_gnn_softmax_qnli/ 2>&1 | tee glue_roberta_moe_gnn_softmax_qnli.out
