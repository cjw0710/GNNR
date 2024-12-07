import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse, add_self_loops
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split

# 检查是否有可用的 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载数据
adj = torch.load('cora_adj_out.pt').to(device)
features = torch.load('cora_feature_out.pt').to(device)
labels = torch.load('cora_labels.pt').to(device)

# 转换邻接矩阵为 edge_index 格式
edge_index, edge_weight = dense_to_sparse(adj)
edge_index = edge_index.to(device)
edge_weight = edge_weight.to(device)

# 添加自连接
edge_index, edge_weight = add_self_loops(edge_index, edge_weight, num_nodes=adj.size(0))
edge_index = edge_index.to(device)
edge_weight = edge_weight.to(device)

# 划分数据集
num_nodes = features.size(0)
indices = torch.arange(num_nodes)

# 70% 训练，15% 验证，15% 测试
train_indices, test_indices = train_test_split(indices.numpy(), test_size=0.30, random_state=42)
val_indices, test_indices = train_test_split(test_indices, test_size=0.50, random_state=42)

train_mask = torch.zeros(num_nodes, dtype=torch.bool)
train_mask[train_indices] = True

val_mask = torch.zeros(num_nodes, dtype=torch.bool)
val_mask[val_indices] = True

test_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask[test_indices] = True

# 打印数据分割信息（可选）
print("Training nodes:", train_mask.sum().item())
print("Validation nodes:", val_mask.sum().item())
print("Test nodes:", test_mask.sum().item())

# 定义 GCN 模型
class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        return F.log_softmax(x, dim=1)

# 初始化模型并将其移动到选定的设备
model = GCN(num_node_features=features.size(1), hidden_channels=16, num_classes=len(labels.unique())).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

# 训练函数
def train():
    model.train()
    optimizer.zero_grad()  # 清空优化器的梯度
    out = model(features, edge_index, edge_weight)
    loss = criterion(out[train_mask], labels[train_mask])
    
    # 打印损失值
    print(f"Loss before backward: {loss.item()}")
    
    loss.backward()  # 反向传播计算梯度
    optimizer.step()  # 更新模型参数

    return loss.item()

# 验证函数
def validate():
    model.eval()
    with torch.no_grad():  # 禁用梯度计算
        out = model(features, edge_index, edge_weight)
        pred = out[val_mask].argmax(dim=1)
        correct = (pred == labels[val_mask]).sum()
        acc = int(correct) / int(labels[val_mask].size(0))
    return acc

# 测试函数
def test():
    model.eval()
    with torch.no_grad():  # 禁用梯度计算
        out = model(features, edge_index, edge_weight)
        pred = out[test_mask].argmax(dim=1)
        correct = (pred == labels[test_mask]).sum()
        acc = int(correct) / int(labels[test_mask].size(0))
    return acc

# 训练模型
for epoch in range(200):
    try:
        loss = train()
        if epoch % 10 == 0:
            val_acc = validate()
            test_acc = test()
            print(f'Epoch {epoch}, Loss: {loss:.4f}, Validation Accuracy: {val_acc:.4f}, Test Accuracy: {test_acc:.4f}')
    except RuntimeError as e:
        print(f'Error during training: {e}')
        break
