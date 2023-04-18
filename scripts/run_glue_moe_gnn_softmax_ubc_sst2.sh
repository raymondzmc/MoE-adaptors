conda activate moe-adaptors
export CUDA_VISIBLE_DEVICES=2

python main.py -plm_name roberta-base -adator_type moe -expert_type gnn -dataset sst2 -gate_type softmax -output_dir output/glue/roberta/moe_gnn_softmax_sst2/ 2>&1 | tee glue_roberta_moe_gnn_softmax_sst2.out
