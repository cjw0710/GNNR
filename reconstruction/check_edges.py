import argparse
import torch
from torch_geometric.datasets import DBLP

# 检查数据集中的边类型
def check_edges(dataset):
    print("Edge types in the dataset:")
    edge_index_dict = dataset.data.edge_index_dict
    for edge_type in edge_index_dict.keys():
        print(edge_type)
        if edge_type not in edge_index_dict:
            print(f"{edge_type} does not exist in edge_index_dict.")

# 参数解析
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, default='./data/DBLP')
args = parser.parse_args()

# 加载数据集
dataset = DBLP(root=args.dataset_path)

# 检查边类型
check_edges(dataset)
