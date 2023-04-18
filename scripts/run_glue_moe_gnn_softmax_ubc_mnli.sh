conda activate moe-adaptors
export CUDA_VISIBLE_DEVICES=2

python main.py -plm_name roberta-base -adator_type moe -expert_type gnn -dataset mnli -gate_type softmax -output_dir output/glue/roberta/moe_gnn_softmax_mnli/ 2>&1 | tee glue_roberta_moe_gnn_softmax_mnli.out
