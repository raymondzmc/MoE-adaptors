conda activate moe-adaptors
export CUDA_VISIBLE_DEVICES=1

# python main.py -plm_name roberta-base -adator_type moe -expert_type gnn -dataset cola -gate_type gumbel -do_eval -eval_all -output_dir output/glue/roberta/moe_gnn_gumbel_cola/ 
# python main.py -plm_name roberta-base -adator_type moe -expert_type gnn -dataset mrpc -gate_type gumbel -do_eval -eval_all -output_dir output/glue/roberta/moe_gnn_gumbel_mrpc/
# python main.py -plm_name roberta-base -adator_type moe -expert_type gnn -dataset rte -gate_type gumbel -do_eval -eval_all -output_dir output/glue/roberta/moe_gnn_gumbel_rte/
# python main.py -plm_name roberta-base -adator_type moe -expert_type gnn -dataset stsb -gate_type gumbel -do_eval -eval_all -output_dir output/glue/roberta/moe_gnn_gumbel_stsb/ 
python main.py -plm_name roberta-base -adator_type moe -expert_type gnn -dataset sst2 -gate_type gumbel -do_eval -eval_all -output_dir output/glue/roberta/moe_gnn_gumbel_sst2/ 

# python main.py -plm_name roberta-base -adator_type moe -expert_type gnn -dataset cola -gate_type softmax -do_eval -eval_all -output_dir output/glue/roberta/moe_gnn_softmax_cola/ 
# python main.py -plm_name roberta-base -adator_type moe -expert_type gnn -dataset mrpc -gate_type softmax -do_eval -eval_all -output_dir output/glue/roberta/moe_gnn_softmax_mrpc/
# python main.py -plm_name roberta-base -adator_type moe -expert_type gnn -dataset rte -gate_type softmax -do_eval -eval_all -output_dir output/glue/roberta/moe_gnn_softmax_rte/
python main.py -plm_name roberta-base -adator_type moe -expert_type gnn -dataset stsb -gate_type softmax -do_eval -eval_all -output_dir output/glue/roberta/moe_gnn_softmax_stsb/
# python main.py -plm_name roberta-base -adator_type moe -expert_type gnn -dataset sst2 -gate_type softmax -do_eval -eval_all -output_dir output/glue/roberta/moe_gnn_softmax_sst2/ 