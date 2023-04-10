export CUDA_VISIBLE_DEVICES=3
python data/parse.py -graph_types syntax -dataset cola -resource_dir ./resources/dgl_graphs
python data/parse.py -graph_types syntax -dataset mrpc -resource_dir ./resources/dgl_graphs
python data/parse.py -graph_types syntax -dataset qnli -resource_dir ./resources/dgl_graphs
python data/parse.py -graph_types syntax -dataset rte -resource_dir ./resources/dgl_graphs
python data/parse.py -graph_types syntax -dataset stsb -resource_dir ./resources/dgl_graphs
