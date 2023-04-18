conda activate moe-adaptors
export CUDA_VISIBLE_DEVICES=1

python main.py -plm_name roberta-base -freeze_plm -pooling_method cls -adator_type moe -expert_type gnn -dataset qqp -gate_type gumbel -output_dir output/glue/roberta/moe_gnn_gumbel_qqp_cls/ 2>&1 | tee glue_roberta_moe_gnn_gumbel_qqp_cls.out
