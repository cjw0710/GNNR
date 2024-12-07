import torch
import numpy as np
import scipy.sparse as sp
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

# 加载数据集的函数
def load_data(dataset_name):
    path = f'./data/{dataset_name}'
    dataset = Planetoid(root=path, name=dataset_name, transform=NormalizeFeatures())
    data = dataset[0]
    features = data.x
    labels = data.y
    
    adj = sp.coo_matrix((np.ones(data.edge_index.shape[1]), (data.edge_index[0], data.edge_index[1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj, features, labels

# # 如果要加载Citeseer数据集
# if dataset == 'citeseer':
# pubmed
adj, features, labels = load_data('citeseer')
class_sample_num = 20
im_class_num = 3


print(adj)
print(features)
print(labels)
