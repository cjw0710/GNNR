# coding=utf-8

import torch
import os
import numpy as np
from sklearn.model_selection import train_test_split
from rphgnn.utils.graph_utils import add_random_feats, dgl_add_all_reversed_edges, dgl_remove_edges
from .hgb import load_imdb, load_freebase, load_dblp, load_hgb_acm
from tqdm import tqdm
import pickle
from gensim.models import Word2Vec
import time
import dgl
# from ogb.nodeproppred import DglNodePropPredDataset
import scipy
import dgl.data
'''
def load_mag(device):
    
    # path = args.use_emb
    home_dir = os.getenv("HOME")
    dataset = DglNodePropPredDataset(
        name="ogbn-mag", root=os.path.join(home_dir, ".ogb", "dataset"))
    g, labels = dataset[0]

    # my
    g = g.to(device)

    splitted_idx = dataset.get_idx_split()
    train_nid = splitted_idx["train"]['paper']
    val_nid = splitted_idx["valid"]['paper']
    test_nid = splitted_idx["test"]['paper']
    features = g.nodes['paper'].data['feat']
    g.nodes["paper"].data["feat"] = features.to(device)


    labels = labels['paper'].to(device).squeeze()
    n_classes = int(labels.max() - labels.min()) + 1
    train_nid, val_nid, test_nid = np.array(train_nid), np.array(val_nid), np.array(test_nid)


    target_node_type = "paper"
    feature_node_types = [target_node_type]

    return g, target_node_type, feature_node_types, labels, n_classes, train_nid, val_nid, test_nid
'''
def load_cornell(device, dataset, data_path="C:\\Users\\皮皮伟\\Desktop\\CL2IGNN\\CLGNN-main\\data"):
    n_classes = 5  # 根据 Cornell 数据集的实际类别数进行调整
    # 加载邻接矩阵并转换为 DGL 图
    adj_matrix = torch.load(os.path.join(data_path, "cornell_adj_matrix.pt"), map_location=torch.device('cpu'))
    graph = dgl.from_scipy(scipy.sparse.coo_matrix(adj_matrix.detach().cpu().numpy()))
    graph = graph.to(device)

    # 加载节点特征
    node_features = torch.load(os.path.join(data_path, "cornell_features.pt"), map_location=torch.device('cpu')).detach().cpu()
    graph.ndata['feat'] = node_features

    # 加载标签
    label_tmp = torch.load(os.path.join(data_path, "cornell_labels.pt"), map_location=torch.device('cpu'))
    labels = torch.from_numpy(label_tmp.detach().cpu().numpy())

    # 存储不同集合的节点索引
    train_nid, val_nid, test_nid = [], [], []

    # 获取所有独特的类标签
    unique_labels = np.unique(labels)

    # 对每个类标签进行划分
    for label in unique_labels:
        # 获取当前类的所有节点索引
        label_indices = np.where(labels == label)[0]

        # 分成50%作为测试集
        train_val_indices, test_indices = train_test_split(label_indices, test_size=0.5, random_state=42)

        # 将剩下的节点再分成25%作为训练集，25%作为验证集
        train_indices, val_indices = train_test_split(train_val_indices, test_size=0.5, random_state=42)

        # 将结果添加到各自的列表中
        train_nid.extend(train_indices)
        val_nid.extend(val_indices)
        test_nid.extend(test_indices)

    # 转换为 NumPy 数组
    train_nid = np.array(train_nid)
    val_nid = np.array(val_nid)
    test_nid = np.array(test_nid)

    # 目标节点类型和特征节点类型（根据实际情况调整）
    target_node_type = "paper"  # 或者根据你的数据集结构调整
    feature_node_types = [target_node_type]

    return graph, target_node_type, feature_node_types, labels, n_classes, train_nid, val_nid, test_nid

# def load_texas(device, dataset, data_path="C:\\Users\\皮皮伟\\Desktop\\CL2IGNN\\CLGNN-main\\data"):
#     n_classes = 5  # 根据 texas 数据集的实际类别数进行调整
#     # 加载邻接矩阵并转换为 DGL 图
#     adj_matrix = torch.load(os.path.join(data_path, "texas_adj_matrix.pt"), map_location=torch.device('cpu'))
#     graph = dgl.from_scipy(scipy.sparse.coo_matrix(adj_matrix.detach().cpu().numpy()))
#     graph = graph.to(device)

#     # 加载节点特征
#     node_features = torch.load(os.path.join(data_path, "texas_features.pt"), map_location=torch.device('cpu')).detach().cpu()
#     graph.ndata['feat'] = node_features

#     # 加载标签
#     label_tmp = torch.load(os.path.join(data_path, "texas_labels.pt"), map_location=torch.device('cpu'))
#     labels = torch.from_numpy(label_tmp.detach().cpu().numpy())

#     # 存储不同集合的节点索引
#     train_nid, val_nid, test_nid = [], [], []

#     # 获取所有独特的类标签
#     unique_labels = np.unique(labels)

#     # 对每个类标签进行划分
#     for label in unique_labels:
#         # 获取当前类的所有节点索引
#         label_indices = np.where(labels == label)[0]

#         # 分成50%作为测试集
#         train_val_indices, test_indices = train_test_split(label_indices, test_size=0.5, random_state=42)

#         # 将剩下的节点再分成25%作为训练集，25%作为验证集
#         train_indices, val_indices = train_test_split(train_val_indices, test_size=0.5, random_state=42)

#         # 将结果添加到各自的列表中
#         train_nid.extend(train_indices)
#         val_nid.extend(val_indices)
#         test_nid.extend(test_indices)

#     # 转换为 NumPy 数组
#     train_nid = np.array(train_nid)
#     val_nid = np.array(val_nid)
#     test_nid = np.array(test_nid)

#     # 目标节点类型和特征节点类型（根据实际情况调整）
#     target_node_type = "paper"  # 或者根据你的数据集结构调整
#     feature_node_types = [target_node_type]

#     return graph, target_node_type, feature_node_types, labels, n_classes, train_nid, val_nid, test_nid
def load_texas(device, dataset, data_path="C:\\Users\\皮皮伟\\Desktop\\CL2IGNN\\CLGNN-main\\data"):
    n_classes = 5  # 根据 texas 数据集的实际类别数进行调整

    # 加载邻接矩阵并转换为 DGL 图
    adj_matrix = torch.load(os.path.join(data_path, "texas_adj_matrix.pt"), map_location=torch.device('cpu'))
    graph = dgl.from_scipy(scipy.sparse.coo_matrix(adj_matrix.detach().cpu().numpy()))
    graph = graph.to(device)

    # 加载节点特征
    node_features = torch.load(os.path.join(data_path, "texas_features.pt"), map_location=torch.device('cpu')).detach().cpu()
    graph.ndata['feat'] = node_features

    # 加载标签
    label_tmp = torch.load(os.path.join(data_path, "texas_labels.pt"), map_location=torch.device('cpu'))
    labels = torch.from_numpy(label_tmp.detach().cpu().numpy())

    # 存储不同集合的节点索引
    train_nid, val_nid, test_nid = [], [], []

    # 获取所有独特的类标签
    unique_labels = np.unique(labels)

    # 对每个类标签进行划分
    for label in unique_labels:
        # 获取当前类的所有节点索引
        label_indices = np.where(labels == label)[0]

        # 确保有足够的样本进行划分
        if len(label_indices) < 5:
            continue  # 跳过样本少于5的类

        # 分成20%作为测试集
        train_val_indices, test_indices = train_test_split(label_indices, test_size=0.2, random_state=42)

        # 将剩下的80%再分成60%作为训练集，20%作为验证集
        train_indices, val_indices = train_test_split(train_val_indices, test_size=0.25, random_state=42)  # 20/80 = 0.25

        # 将结果添加到各自的列表中
        train_nid.extend(train_indices)
        val_nid.extend(val_indices)
        test_nid.extend(test_indices)

    # 转换为 NumPy 数组
    train_nid = np.array(train_nid)
    val_nid = np.array(val_nid)
    test_nid = np.array(test_nid)
    print("训练集标签顺序:", labels[train_nid])
    print("验证集标签顺序:", labels[val_nid])
    print("测试集标签顺序:", labels[test_nid])
    # 目标节点类型和特征节点类型（根据实际情况调整）
    target_node_type = "paper"  # 或者根据你的数据集结构调整
    feature_node_types = [target_node_type]
    # 在函数的最后添加以下代码
    test_labels = labels[test_nid]
    print("测试集中的节点类别：", test_labels.numpy())

    return graph, target_node_type, feature_node_types, labels, n_classes, train_nid, val_nid, test_nid
def load_cora(device, dataset, data_path="C:\\Users\\皮皮伟\\Desktop\\CL2IGNN\\CLGNN-main\\data"):
    n_classes = 7
    # dataset = dgl.data.CoraGraphDataset()
    # graph = dataset[0]
    adj_matrix = torch.load(os.path.join(data_path,"cora_adj_out.pt"),map_location=torch.device('cpu'))
    graph = dgl.from_scipy(scipy.sparse.coo_matrix(adj_matrix.detach().cpu().numpy()))
    graph = graph.to(device)
    node_features = torch.load(os.path.join(data_path,"cora_feature_out.pt"),map_location=torch.device('cpu')).detach().cpu()
    graph.ndata['feat'] = node_features
    label_tmp = torch.load(os.path.join(data_path, "cora_labels.pt"), map_location=torch.device('cpu'))
    labels = torch.from_numpy(label_tmp.detach().cpu().numpy())
    # 存储不同集的节点索引
    train_nid, val_nid, test_nid = [], [], []

    # 获取所有独特的类标签
    unique_labels = np.unique(labels)

    # 对每个类标签进行划分
    for label in unique_labels:
        # 获取当前类的所有节点索引
        label_indices = np.where(labels == label)[0]
        
        # 分成50%作为测试集
        train_val_indices, test_indices = train_test_split(label_indices, test_size=0.5, random_state=42)
        
        # 将剩下的节点再分成25%作为训练集，25%作为验证集
        train_indices, val_indices = train_test_split(train_val_indices, test_size=0.5, random_state=42)
        
        # 将结果添加到各自的列表中
        train_nid.extend(train_indices)
        val_nid.extend(val_indices)
        test_nid.extend(test_indices)
    print(train_nid)
    print(test_nid)
    # 转换为NumPy数组
    train_nid = np.array(train_nid)
    val_nid = np.array(val_nid)
    test_nid = np.array(test_nid)
    # 
   
   
    #2738
    target_node_type = "paper"
    feature_node_types = [target_node_type]
    return graph, target_node_type, feature_node_types, labels, n_classes, train_nid, val_nid, test_nid


def load_BlogCatalog(device, dataset, data_path="C:\\Users\\皮皮伟\\Desktop\\CL2IGNN\\CLGNN-main\\data"):
    n_classes = 36
    # dataset = dgl.data.CoraGraphDataset()
    # graph = dataset[0]
    adj_matrix = torch.load(os.path.join(data_path,"BlogCatalog_adj_out.pt"),map_location=torch.device('cpu'))
    print(adj_matrix.shape)
    graph = dgl.from_scipy(scipy.sparse.coo_matrix(adj_matrix.detach().cpu().numpy()))
    graph = graph.to(device)
    node_features = torch.load(os.path.join(data_path,"BlogCatalog_feature_out.pt"),map_location=torch.device('cpu')).detach().cpu()
    print(node_features.shape)
    graph.ndata['feat'] = node_features
    label_tmp = torch.load(os.path.join(data_path, "BlogCatalog_labels.pt"), map_location=torch.device('cpu'))
    labels = torch.from_numpy(label_tmp.detach().cpu().numpy())
    print(labels)
    
    # 存储不同集的节点索引
    train_nid, val_nid, test_nid = [], [], []

    # 获取所有独特的类标签
    unique_labels = np.unique(labels)

    # 对每个类标签进行划分
    for label in unique_labels:
        # 获取当前类的所有节点索引
        label_indices = np.where(labels == label)[0]
        
        # 分成50%作为测试集
        train_val_indices, test_indices = train_test_split(label_indices, test_size=0.5, random_state=42)
        
        # 将剩下的节点再分成25%作为训练集，25%作为验证集
        train_indices, val_indices = train_test_split(train_val_indices, test_size=0.5, random_state=42)
        # # 第一阶段：70% 训练集，30% 测试集
        # train_val_indices, test_indices = train_test_split(label_indices, test_size=0.20, random_state=42)

        # # 第二阶段：将剩余的30%分成15%验证集和15%测试集
        # train_indices, val_indices = train_test_split(train_val_indices, test_size=0.50, random_state=42)
        # 将结果添加到各自的列表中
        train_nid.extend(train_indices)
        val_nid.extend(val_indices)
        test_nid.extend(test_indices)

    # 转换为NumPy数组
    train_nid = np.array(train_nid)
    val_nid = np.array(val_nid)
    test_nid = np.array(test_nid)
    #2738
    target_node_type = "paper"
    feature_node_types = [target_node_type]
    return graph, target_node_type, feature_node_types, labels, n_classes, train_nid, val_nid, test_nid

def load_citeseer(device, dataset, data_path="C:\\Users\\皮皮伟\\Desktop\\CL2IGNN\\CLGNN-main\\data"):
    n_classes = 6
    # dataset = dgl.data.CoraGraphDataset()
    # graph = dataset[0]
    adj_matrix = torch.load(os.path.join(data_path,"citeseer_generated_G.pt"),map_location=torch.device('cpu'))
    print(adj_matrix.shape)
    graph = dgl.from_scipy(scipy.sparse.coo_matrix(adj_matrix.detach().cpu().numpy()))
    graph = graph.to(device)
    node_features = torch.load(os.path.join(data_path,"citeseer_feature_out.pt"),map_location=torch.device('cpu')).detach().cpu()
    graph.ndata['feat'] = node_features
    label_tmp = torch.load(os.path.join(data_path, "citeseer_labels_new.pt"), map_location=torch.device('cpu'))
    labels = torch.from_numpy(label_tmp.detach().cpu().numpy())
    
    # 存储不同集的节点索引
    train_nid, val_nid, test_nid = [], [], []

    # 获取所有独特的类标签
    unique_labels = np.unique(labels)

    # 对每个类标签进行划分
    for label in unique_labels:
        # 获取当前类的所有节点索引
        label_indices = np.where(labels == label)[0]
        
        # 分成50%作为测试集
        train_val_indices, test_indices = train_test_split(label_indices, test_size=0.5, random_state=42)
        
        # 将剩下的节点再分成25%作为训练集，25%作为验证集
        train_indices, val_indices = train_test_split(train_val_indices, test_size=0.5, random_state=42)
        
        # 将结果添加到各自的列表中
        train_nid.extend(train_indices)
        val_nid.extend(val_indices)
        test_nid.extend(test_indices)

    print(train_indices)
    print(test_indices)
    # 转换为NumPy数组
    train_nid = np.array(train_nid)
    val_nid = np.array(val_nid)
    test_nid = np.array(test_nid)
    #2738
    target_node_type = "paper"
    feature_node_types = [target_node_type]
    return graph, target_node_type, feature_node_types, labels, n_classes, train_nid, val_nid, test_nid



def load_wiki(device, dataset, data_path="C:\\Users\\皮皮伟\\Desktop\\CL2IGNN\\CLGNN-main\\data"):
    n_classes = 6
    # dataset = dgl.data.CoraGraphDataset()
    # graph = dataset[0]
    adj_matrix = torch.load(os.path.join(data_path,"wiki-cs_adj_out.pt"),map_location=torch.device('cpu'))
    print(adj_matrix.shape)
    graph = dgl.from_scipy(scipy.sparse.coo_matrix(adj_matrix.detach().cpu().numpy()))
    graph = graph.to(device)
    node_features = torch.load(os.path.join(data_path,"wiki-cs_feature_out.pt"),map_location=torch.device('cpu')).detach().cpu()
    graph.ndata['feat'] = node_features
    label_tmp = torch.load(os.path.join(data_path, "wiki-cs_labels.pt"), map_location=torch.device('cpu'))
    labels = torch.from_numpy(label_tmp.detach().cpu().numpy())
    
    # 存储不同集的节点索引
    train_nid, val_nid, test_nid = [], [], []

    # 获取所有独特的类标签
    unique_labels = np.unique(labels)

    # 对每个类标签进行划分
    for label in unique_labels:
        # 获取当前类的所有节点索引
        label_indices = np.where(labels == label)[0]
        
        # 分成50%作为测试集
        train_val_indices, test_indices = train_test_split(label_indices, test_size=0.5, random_state=42)
        
        # 将剩下的节点再分成25%作为训练集，25%作为验证集
        train_indices, val_indices = train_test_split(train_val_indices, test_size=0.5, random_state=42)
        
        # 将结果添加到各自的列表中
        train_nid.extend(train_indices)
        val_nid.extend(val_indices)
        test_nid.extend(test_indices)

    # 转换为NumPy数组
    train_nid = np.array(train_nid)
    val_nid = np.array(val_nid)
    test_nid = np.array(test_nid)
    #2738
    target_node_type = "paper"
    feature_node_types = [target_node_type]
    return graph, target_node_type, feature_node_types, labels, n_classes, train_nid, val_nid, test_nid

def load_dgl_citeseer(embedding_size):
    device = "cpu"
    g, target_node_type, feature_node_types, labels, n_classes, train_index, valid_index, test_index = load_citeseer(device, "citeseer")

    # g.nodes[target_node_type].data["label"] = labels
    g.ndata['node_label'] = labels
    #g = add_random_feats(g, embedding_size, excluded_ntypes=feature_node_types)

    return g, target_node_type, feature_node_types, (train_index, valid_index, test_index)

def load_dgl_cornell(embedding_size):
    device = "cpu"
    g, target_node_type, feature_node_types, labels, n_classes, train_index, valid_index, test_index = load_cornell(device, "cornell")

    # g.nodes[target_node_type].data["label"] = labels
    g.ndata['node_label'] = labels
    #g = add_random_feats(g, embedding_size, excluded_ntypes=feature_node_types)

    return g, target_node_type, feature_node_types, (train_index, valid_index, test_index)

def load_dgl_texas(embedding_size):
    device = "cpu"
    g, target_node_type, feature_node_types, labels, n_classes, train_index, valid_index, test_index = load_texas(device, "texas")

    # g.nodes[target_node_type].data["label"] = labels
    g.ndata['node_label'] = labels
    #g = add_random_feats(g, embedding_size, excluded_ntypes=feature_node_types)

    return g, target_node_type, feature_node_types, (train_index, valid_index, test_index)

def load_dgl_cora(embedding_size):
    device = "cpu"
    g, target_node_type, feature_node_types, labels, n_classes, train_index, valid_index, test_index = load_cora(device, "cora")

    # g.nodes[target_node_type].data["label"] = labels
    g.ndata['node_label'] = labels
    #g = add_random_feats(g, embedding_size, excluded_ntypes=feature_node_types)

    return g, target_node_type, feature_node_types, (train_index, valid_index, test_index)


def load_dgl_wiki(embedding_size):
    device = "cpu"
    g, target_node_type, feature_node_types, labels, n_classes, train_index, valid_index, test_index = load_wiki(device, "wiki")

    # g.nodes[target_node_type].data["label"] = labels
    g.ndata['node_label'] = labels
    #g = add_random_feats(g, embedding_size, excluded_ntypes=feature_node_types)

    return g, target_node_type, feature_node_types, (train_index, valid_index, test_index)

def load_dgl_BlogCatalog(embedding_size):
    device = "cpu"
    g, target_node_type, feature_node_types, labels, n_classes, train_index, valid_index, test_index = load_BlogCatalog(device, "BlogCatalog")

    # g.nodes[target_node_type].data["label"] = labels
    g.ndata['node_label'] = labels
    #g = add_random_feats(g, embedding_size, excluded_ntypes=feature_node_types)

    return g, target_node_type, feature_node_types, (train_index, valid_index, test_index)

def load_dgl_mag(embedding_size):
    device = "cpu"
    
    g, target_node_type, feature_node_types, labels, n_classes, train_index, valid_index, test_index = load_mag(device)

    g.nodes[target_node_type].data["label"] = labels


    # embedding_size = g.ndata["feat"][target_node_type].size(-1) * 4  
    g = add_random_feats(g, embedding_size, excluded_ntypes=feature_node_types)

    return g, target_node_type, feature_node_types, (train_index, valid_index, test_index)

def load_dgl_hgb(dataset, use_all_feat=False, embedding_size=None, random_state=None):

    if dataset == "imdb":
        load_func = load_imdb
    elif dataset == "dblp":
        load_func = load_dblp
    elif dataset == "hgb_acm":
        load_func = load_hgb_acm
    elif dataset == "freebase":
        load_func = load_freebase
    else:
        raise RuntimeError(f"Unsupported dataset {dataset}")

    # dgl_graph, target_node_type, feature_node_types, features, features_dict, labels, num_classes, train_indices, valid_indices, test_indices, train_mask, valid_mask, test_mask = load_func(random_state=random_state)

    dgl_graph, target_node_type, feature_node_types, features, features_dict, labels, _, train_indices, valid_indices, test_indices, _, _, _ = load_func(random_state=random_state)


    if use_all_feat:
        print("use all features ...")
        for int_ntype, value in features_dict.items():
            ntype = str(int_ntype)
            if value is None:
                print("skip None ntype: ", ntype)
            else:
                
                print("set feature for ntype: ", ntype, dgl_graph.num_nodes(ntype), value.shape)
                dgl_graph.nodes[ntype].data["feat"] = torch.tensor(value).to(torch.float32)

        if embedding_size is None:
                embedding_size = features.size(-1)

        dgl_graph = add_random_feats(dgl_graph, embedding_size, 
            excluded_ntypes=[ntype for ntype in dgl_graph.ntypes if "feat" in dgl_graph.nodes[ntype].data]
        )

    else:
        if len(feature_node_types) == 0:
            dgl_graph = add_random_feats(dgl_graph, embedding_size, excluded_ntypes=None)
        else:
            dgl_graph.nodes[target_node_type].data["feat"] = features
            if embedding_size is None:
                embedding_size = features.size(-1)

            dgl_graph = add_random_feats(dgl_graph, embedding_size, 
                excluded_ntypes=[ntype for ntype in dgl_graph.ntypes if "feat" in dgl_graph.nodes[ntype].data]
            )
        
    dgl_graph.nodes[target_node_type].data["label"] = labels

    return dgl_graph, target_node_type, feature_node_types, (train_indices, valid_indices, test_indices)

def load_dgl_hgb_acm(use_all_feat=False, embedding_size=None, random_state=None):
    return load_dgl_hgb("hgb_acm", use_all_feat=use_all_feat, embedding_size=embedding_size, random_state=random_state)

def load_dgl_imdb(use_all_feat=False, embedding_size=None, random_state=None):
    return load_dgl_hgb("imdb", use_all_feat=use_all_feat, embedding_size=embedding_size, random_state=random_state)

def load_dgl_dblp(use_all_feat=False, embedding_size=None, random_state=None):
    return load_dgl_hgb("dblp", use_all_feat=use_all_feat, embedding_size=embedding_size, random_state=random_state)

def load_dgl_freebase(use_all_feat=False, embedding_size=None, random_state=None):
    return load_dgl_hgb("freebase", use_all_feat=use_all_feat, embedding_size=embedding_size, random_state=random_state)

def load_oag(device, dataset, data_path="datasets/nars_academic_oag"):
    import pickle
    # assert args.data_dir is not None


    if dataset == "oag_L1":
        graph_file = "graph_L1.pk"
        predict_venue = False
    elif dataset == "oag_venue":
        graph_file = "graph_venue.pk"
        predict_venue = True
    else:
        raise RuntimeError(f"Unsupported dataset {dataset}")
    with open(os.path.join(data_path, graph_file), "rb") as f:
        dataset = pickle.load(f)
    n_classes = dataset["n_classes"]
    graph = dgl.heterograph(dataset["edges"])
    graph = graph.to(device)
    train_nid, val_nid, test_nid = dataset["split"]


    with open(os.path.join(data_path, "paper.npy"), "rb") as f:
        # loading lang features of paper provided by HGT author
        paper_feat = torch.from_numpy(np.load(f)).float().to(device)
    graph.nodes["paper"].data["feat"] = paper_feat[:graph.number_of_nodes("paper")]

    if predict_venue:
        labels = torch.from_numpy(dataset["labels"])
    else:
        labels = torch.zeros(graph.number_of_nodes("paper"), n_classes)
        for key in dataset["labels"]:
            labels[key, dataset["labels"][key]] = 1
    train_nid, val_nid, test_nid = np.array(train_nid), np.array(val_nid), np.array(test_nid)

    # return graph, labels, n_classes, train_nid, val_nid, test_nid

    target_node_type = "paper"
    feature_node_types = [target_node_type]

    return graph, target_node_type, feature_node_types, labels, n_classes, train_nid, val_nid, test_nid

def load_dgl_oag(dataset, data_path="datasets/nars_academic_oag", embedding_size=None):
    g, target_node_type, feature_node_types, labels, n_classes, train_index, valid_index, test_index = load_oag(device="cpu", dataset=dataset, data_path=data_path)

    target_node_type = "paper"

    g = add_random_feats(g, embedding_size, excluded_ntypes=[target_node_type])
    
    g.nodes[target_node_type].data["label"] = labels


    return g, target_node_type, feature_node_types, (train_index, valid_index, test_index)
    # return dgl_graph, target_node_type, (train_index, valid_index, test_index)
    
def nrl_update_features(dataset, hetero_graph, excluded_ntypes, 
                        nrl_pretrain_epochs=40, embedding_size=512):
 
    start_time = time.time()
    nrl_cache_path = os.path.join("./cache/{}.p".format(dataset))

    if os.path.exists(nrl_cache_path):
        print("loading cache: {}".format(nrl_cache_path))
        with open(nrl_cache_path, "rb") as f:
            nrl_embedding_dict = pickle.load(f)
    else:
        
        vocab_corpus = []
        for ntype in hetero_graph.ntypes:
            for i in tqdm(range(hetero_graph.num_nodes(ntype))):
                vocab_corpus.append(["{}_{}".format(ntype, i)])

        
        corpus = []
        for etype in hetero_graph.canonical_etypes:
            if etype[1].startswith("r."):
                print("skip etype: ", etype)
                continue
            row, col = hetero_graph.edges(etype=etype)
            for i, j in tqdm(zip(row, col)):
                corpus.append(["{}_{}".format(etype[0], i), "{}_{}".format(etype[2], j)])

        print("start training word2vec")
        # word2vec_model = Word2Vec(sentences=vocab_corpus, vector_size=embedding_size, window=2, min_count=0, workers=4)
        ###############################################################################################################
        word2vec_model = Word2Vec(sentences=vocab_corpus, vector_size=embedding_size, window=2, min_count=0, workers=4)
        for i in tqdm(range(nrl_pretrain_epochs)):
            print("train word2vec epoch {}".format(i))
            word2vec_model.train(corpus, total_examples=len(corpus), epochs=1)
        ################################################################################################################
        # word2vec_model = Word2Vec(sentences=vocab_corpus, vector_size=embedding_size, window=2, min_count=0, workers=4, negative=20)
        
        # print("train word2vec ...")
        # word2vec_model.train(corpus, total_examples=len(corpus), epochs=nrl_pretrain_epochs)

        nrl_embedding_dict = {}
        for ntype in hetero_graph.ntypes:
            embeddings = np.array([word2vec_model.wv["{}_{}".format(ntype, i)] for i in range(hetero_graph.num_nodes(ntype))])
            nrl_embedding_dict[ntype] = embeddings
            

        print("saving cache: {}".format(nrl_cache_path))
        with open(nrl_cache_path, "wb") as f:
            pickle.dump(nrl_embedding_dict, f, protocol=4)
    


    print("nrl time: ", time.time() - start_time)


    for ntype in list(hetero_graph.ntypes):
        if ntype not in excluded_ntypes:
            print("using NRL embeddings for featureless nodetype: {}".format(ntype))
            # hetero_graph.x_dict[node_type] = nrl_embedding_dict[node_type]
            print("!!!!!!!!!!!!!!!!!!!!")
            hetero_graph.nodes[ntype].data["feat"] = torch.tensor(nrl_embedding_dict[ntype])

    return hetero_graph

def load_dgl_data(dataset, use_all_feat=False, embedding_size=None, use_nrl=False, random_state=None):


    batch_size = 10000
    num_epochs = 510
    patience = 30
    validation_freq = 10
    convert_to_tensor = True


    if dataset == "mag":
        hetero_graph, target_node_type, feature_node_types, (train_index, valid_index, test_index) = load_dgl_mag(embedding_size=embedding_size)        
        convert_to_tensor = False
        num_epochs = 100
        patience = 10

    elif dataset in ["oag_L1", "oag_venue"]:

        batch_size = 3000
        if embedding_size is None:
            embedding_size = 128 * 2

        hetero_graph, target_node_type, feature_node_types, (train_index, valid_index, test_index) = load_dgl_oag(dataset, embedding_size=embedding_size)

        convert_to_tensor = False

        num_epochs = 200
        patience = 10

    elif dataset == "imdb":
        
        if embedding_size is None:
            embedding_size = 1024

        hetero_graph, target_node_type, feature_node_types, (train_index, valid_index, test_index) = load_dgl_imdb(use_all_feat=use_all_feat, 
            embedding_size=embedding_size, random_state=random_state)

        etypes_to_remove = set()
        for etype in hetero_graph.canonical_etypes:
            etype_ = etype[1]
            items = list(etype_)
            print("items: ", items)
            if items[0] > items[1]:
                etypes_to_remove.add(etype)
                print("remove items: ", items)

        print("etypes_to_remove: ", etypes_to_remove)

        hetero_graph = dgl_remove_edges(hetero_graph, etypes_to_remove)
        print("remaining etypes: ", hetero_graph.canonical_etypes)

        num_epochs = 500
        patience = 200

        validation_freq = 1

    elif dataset == "dblp":

        if embedding_size is None:
            embedding_size = 1024
        hetero_graph, target_node_type, feature_node_types, (train_index, valid_index, test_index) = load_dgl_dblp(use_all_feat=use_all_feat, embedding_size=embedding_size, random_state=random_state)
        # hetero_graph, target_node_type, feature_node_types, (train_index, valid_index, test_index) = load_dgl_dblp(embedding_size=256)

        print("raw etypes: ", hetero_graph.canonical_etypes)
        etypes_to_remove = set()
        for etype in hetero_graph.canonical_etypes:
            etype_ = etype[1]
            items = list(etype_)
            print("items: ", items)
            if items[0] > items[1]:
                etypes_to_remove.add(etype)
                print("remove items: ", items)

        print("etypes_to_remove: ", etypes_to_remove)

        hetero_graph = dgl_remove_edges(hetero_graph, etypes_to_remove)

        print("remaining etypes: ", hetero_graph.canonical_etypes)

        # hetero_graph = dgl_add_duplicated_edges(hetero_graph, 3)
        # print("edges update duplication: ", hetero_graph.canonical_etypes)

        num_epochs = 500
        patience = 30
        # validation_freq = 1
        

        # hetero_graph = hetero_graph.add_reversed_edges(inplace=True)

    elif dataset == "hgb_acm":

        if embedding_size is None:
            embedding_size = 512

        hetero_graph, target_node_type, feature_node_types, (train_index, valid_index, test_index) = load_dgl_hgb_acm(use_all_feat=use_all_feat, embedding_size=embedding_size, random_state=random_state)
        # hetero_graph = hetero_graph.add_reversed_edges(inplace=True)
        
        num_epochs = 100
        patience = 20

        validation_freq = 1
        batch_size = 1000

        # for etype in hetero_graph.etypes:
        #     print(etype)
        etypes_to_remove = set()
        for etype in hetero_graph.canonical_etypes:
            etype_ = etype[1]
            items = list(etype_)
            print("items: ", items)
            if etype_[0] == "-" or items[0] > items[1]:
                etypes_to_remove.add(etype)
                print("remove items: ", items)

        print("etypes_to_remove: ", etypes_to_remove)

        hetero_graph = dgl_remove_edges(hetero_graph, etypes_to_remove)
        print("remaining etypes: ", hetero_graph.canonical_etypes)
        
    elif dataset == "freebase":
        num_epochs = 200
        patience = 20
        # validation_freq = 1
        # hetero_graph, target_node_type, (train_index, valid_index, test_index) = load_dgl_freebase(embedding_size=128)
        if embedding_size is None:
            embedding_size = 512
        hetero_graph, target_node_type, feature_node_types, (train_index, valid_index, test_index) = load_dgl_freebase(use_all_feat=use_all_feat, embedding_size=embedding_size, random_state=random_state)



        etypes_to_remove = set()
        for etype in hetero_graph.canonical_etypes:
            etype_ = etype[1]
            items = [int(c) for c in list(etype_)]
            print("items: ", items)
            if items[0] > items[1]:
                etypes_to_remove.add(etype)
                print("remove items: ", items)

        print("etypes_to_remove: ", etypes_to_remove)

        hetero_graph = dgl_remove_edges(hetero_graph, etypes_to_remove)

        print("etypes: ", hetero_graph.canonical_etypes)

    elif dataset == "cora":  # samples is 2738
        if embedding_size is None:
            embedding_size = 64
        hetero_graph, target_node_type, feature_node_types, (train_index, valid_index, test_index) = load_dgl_cora(embedding_size)        

        convert_to_tensor = False
        num_epochs = 2
        patience = 10
        
    elif dataset == "citeseer":  # samples is 2738
        if embedding_size is None:
            embedding_size = 64
        hetero_graph, target_node_type, feature_node_types, (train_index, valid_index, test_index) = load_dgl_citeseer(embedding_size)        

        convert_to_tensor = False
        num_epochs = 2
        patience = 10
        
    elif dataset == "BlogCatalog":  # samples is 2738
        if embedding_size is None:
            embedding_size = 64
        hetero_graph, target_node_type, feature_node_types, (train_index, valid_index, test_index) = load_dgl_BlogCatalog(embedding_size)        

        convert_to_tensor = False
        num_epochs = 2
        patience = 10
        
    elif dataset == "cornell":  
        if embedding_size is None:
            embedding_size = 64
        hetero_graph, target_node_type, feature_node_types, (train_index, valid_index, test_index) = load_dgl_cornell(embedding_size)        

        convert_to_tensor = False
        num_epochs = 2
        patience = 10   
    elif dataset == "texas":  
        if embedding_size is None:
            embedding_size = 64
        hetero_graph, target_node_type, feature_node_types, (train_index, valid_index, test_index) = load_dgl_texas(embedding_size)        

        convert_to_tensor = False
        num_epochs = 2
        patience = 10       
    elif dataset == "wiki":  # samples is 2738
        if embedding_size is None:
            embedding_size = 64
        hetero_graph, target_node_type, feature_node_types, (train_index, valid_index, test_index) = load_dgl_wiki(embedding_size)        

        convert_to_tensor = False
        num_epochs = 2
        patience = 10
    # hetero_graph = dgl_add_label_nodes(hetero_graph, target_node_type, train_index)
    hetero_graph = dgl.add_reverse_edges(hetero_graph, ignore_bipartite=True)
    hetero_graph = dgl_add_all_reversed_edges(hetero_graph)

    if use_nrl:

        if dataset == "freebase":
            excluded_ntypes = []
        else:
            excluded_ntypes = [target_node_type]


        hetero_graph = nrl_update_features(dataset, hetero_graph, excluded_ntypes)


    return hetero_graph, target_node_type, feature_node_types, (train_index, valid_index, test_index), \
           batch_size, num_epochs, patience, validation_freq, convert_to_tensor

