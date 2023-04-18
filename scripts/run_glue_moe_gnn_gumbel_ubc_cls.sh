conda activate moe-adaptors
export CUDA_VISIBLE_DEVICES=3

python main.py -plm_name roberta-base -freeze_plm -pooling_method cls -adator_type moe -expert_type gnn -dataset cola -gate_type gumbel -output_dir output/glue/roberta/moe_gnn_gumbel_cola/ -plot_gates 2>&1 | tee glue_roberta_moe_gnn_gumbel_cola_cls.out
python main.py -plm_name roberta-base -freeze_plm -pooling_method cls -adator_type moe -expert_type gnn -dataset mrpc -gate_type gumbel -output_dir output/glue/roberta/moe_gnn_gumbel_mrpc/ -plot_gates 2>&1 | tee glue_roberta_moe_gnn_gumbel_mrpc_cls.out
python main.py -plm_name roberta-base -freeze_plm -pooling_method cls -adator_type moe -expert_type gnn -dataset rte -gate_type gumbel -output_dir output/glue/roberta/moe_gnn_gumbel_rte/ -plot_gates 2>&1 | tee glue_roberta_moe_gnn_gumbel_rte_cls.out
python main.py -plm_name roberta-base -freeze_plm -pooling_method cls -adator_type moe -expert_type gnn -dataset stsb -gate_type gumbel -output_dir output/glue/roberta/moe_gnn_gumbel_stsb/ -plot_gates 2>&1 | tee glue_roberta_moe_gnn_gumbel_stsb_cls.out
python main.py -plm_name roberta-base -freeze_plm -pooling_method cls -adator_type moe -expert_type gnn -dataset sst2 -gate_type gumbel -output_dir output/glue/roberta/moe_gnn_gumbel_sst2/ -plot_gates 2>&1 | tee glue_roberta_moe_gnn_gumbel_sst2_cls.out
python main.py -plm_name roberta-base -freeze_plm -pooling_method cls -adator_type moe -expert_type gnn -dataset qnli -gate_type gumbel -output_dir output/glue/roberta/moe_gnn_gumbel_qnli/ -plot_gates 2>&1 | tee glue_roberta_moe_gnn_gumbel_qnli_cls.out
python main.py -plm_name roberta-base -freeze_plm -pooling_method cls -adator_type moe -expert_type gnn -dataset mnli -gate_type gumbel -output_dir output/glue/roberta/moe_gnn_gumbel_mnli_cls/ -plot_gates 2>&1 | tee glue_roberta_moe_gnn_gumbel_mnli_cls.out
python main.py -plm_name roberta-base -freeze_plm -pooling_method cls -adator_type moe -expert_type gnn -dataset qqp -gate_type gumbel -output_dir output/glue/roberta/moe_gnn_gumbel_qqp_cls/ -plot_gates 2>&1 | tee glue_roberta_moe_gnn_gumbel_qqp_cls.out
