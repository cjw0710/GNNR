import torch
import dgl
import numpy as np

# 1. 数据加载
dataset = dgl.data.WisconsinDataset()
graph = dataset[0]

# 2. 加载节点嵌入
embeddings = torch.load('Wisconsinembeds4.pt', map_location=torch.device('cpu'))

# 3. 计算相似性并添加边
threshold = 0.72
num_nodes = graph.number_of_nodes()

for i in range(num_nodes):
    for j in range(i + 1, num_nodes):
        similarity = torch.cosine_similarity(embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0))
        if similarity > threshold:
            graph.add_edges(i, j)
            graph.add_edges(j, i)  # 无向图需要添加反向边

# 4. 添加自环
graph = dgl.add_self_loop(graph)

# 保存邻接矩阵
adj_matrix = graph.adj().to_dense()
torch.save(adj_matrix, 'Wisconsin_adj_matrix.pt')

# 5. 保存特征矩阵
features = graph.ndata['feat']
torch.save(features, 'Wisconsin_features.pt')

# 6. 保存标签矩阵
labels = graph.ndata['label']
torch.save(labels, 'Wisconsin_labels.pt')


# 到这里停止，不执行模型训练和评估部分

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import dgl
# import dgl.data
# import dgl.nn.pytorch as dglnn
# import numpy as np
# from sklearn.metrics import accuracy_score

# # 1. 数据加载
# dataset = dgl.data.CornellDataset()
# graph = dataset[0]

# # 2. 加载节点嵌入
# embeddings = torch.load('Cornellembeds4.pt')

# # 3. 计算相似性并添加边
# threshold = 0.72
# num_nodes = graph.number_of_nodes()
# adj_matrix = np.zeros((num_nodes, num_nodes))

# for i in range(num_nodes):
#     for j in range(i + 1, num_nodes):
#         similarity = torch.cosine_similarity(embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0))
#         if similarity > threshold:
#             graph.add_edges(i, j)
#             graph.add_edges(j, i)  # 无向图需要添加反向边

# # 4. 添加自环
# graph = dgl.add_self_loop(graph)

# adj_matrix = graph.adj().to_dense()  # Convert the adjacency matrix to dense format
# torch.save(adj_matrix, 'cornell_adj_matrix.pt')

# # Feature matrix
# features = graph.ndata['feat']
# torch.save(features, 'cornell_features.pt')

# # Label matrix
# labels = graph.ndata['label']
# torch.save(labels, 'cornell_labels.pt')

# # 检查数据集信息
# print(f"Number of nodes: {graph.number_of_nodes()}")
# print(f"Number of edges: {graph.number_of_edges()}")
# print(f"Node features shape: {graph.ndata['feat'].shape}")
# print(f"Number of classes: {dataset.num_classes}")
# # 5. 数据预处理
# features = graph.ndata['feat']
# labels = graph.ndata['label']
# train_mask = graph.ndata['train_mask'][:, 0].bool()
# val_mask = graph.ndata['val_mask'][:, 0].bool()
# test_mask = graph.ndata['test_mask'][:, 0].bool()



# # 6. 模型构建
# class GCN(nn.Module):
#     def __init__(self, in_feats, h_feats, num_classes):
#         super(GCN, self).__init__()
#         self.conv1 = dglnn.GraphConv(in_feats, h_feats, allow_zero_in_degree=True)
#         self.conv2 = dglnn.GraphConv(h_feats, num_classes, allow_zero_in_degree=True)

#     def forward(self, g, in_feat):
#         h = self.conv1(g, in_feat)
#         h = torch.relu(h)
#         h = self.conv2(g, h)
#         return h

# model = GCN(in_feats=features.shape[1], h_feats=256, num_classes=dataset.num_classes)

# # 7. 模型训练
# optimizer = optim.Adam(model.parameters(), lr=0.01)
# loss_fn = nn.CrossEntropyLoss()
# print(graph)
# for epoch in range(200):
#     model.train()
#     logits = model(graph, features)
#     loss = loss_fn(logits[train_mask], labels[train_mask])
    
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
    
#     if epoch % 10 == 0:
#         train_acc = accuracy_score(labels[train_mask].cpu(), logits[train_mask].argmax(dim=1).cpu())
#         val_acc = accuracy_score(labels[val_mask].cpu(), logits[val_mask].argmax(dim=1).cpu())
#         print(f"Epoch {epoch:05d} | Loss {loss.item():.4f} | Train Acc {train_acc:.4f} | Val Acc {val_acc:.4f}")

# # 8. 模型评估
# model.eval()
# with torch.no_grad():
#     logits = model(graph, features)
#     test_acc = accuracy_score(labels[test_mask].cpu(), logits[test_mask].argmax(dim=1).cpu())
#     print(f"Test Accuracy: {test_acc:.4f}")
