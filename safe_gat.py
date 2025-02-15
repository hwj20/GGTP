import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

# 定义 Safe-GAT 模型
class SafeGAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SafeGAT, self).__init__()
        self.conv1 = GATConv(in_channels, 8, heads=4)
        self.conv2 = GATConv(8 * 4, out_channels, heads=1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# 构建 Torch Geometric 图数据
def graph_to_torch_data(G):
    node_list = list(G.nodes)
    node_features = torch.rand(len(node_list), 4)  # 随机初始化节点特征
    edge_index = torch.tensor([[node_list.index(e[0]), node_list.index(e[1])] for e in G.edges], dtype=torch.long).t().contiguous()
    
    return Data(x=node_features, edge_index=edge_index)

def generate_safety_recommendations(GNN_output, graph):
    attention_nodes = []
    risk_relations = {}

    for i, (node, attr) in enumerate(graph.nodes(data=True)):
        attention_score = GNN_output[i][0].item()
        if attention_score > 0.5:  # 设定注意力阈值
            attention_nodes.append(node)
            for neighbor in graph.neighbors(node):
                risk_relations[f"{node} → {neighbor}"] = f"⚠️ {attr['risk']} 相关风险"

    return {
        "high_attention_nodes": attention_nodes,
        "risk_relations": risk_relations
    }



