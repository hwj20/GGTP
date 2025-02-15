from torch_geometric.nn import TransformerConv
import torch
from torch_geometric.data import Data

# 类似 Graphormer 的 GNN 层
class Graphormer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Graphormer, self).__init__()
        self.conv1 = TransformerConv(in_channels, 128, heads=4, dropout=0.1)
        self.conv2 = TransformerConv(128, out_channels, heads=2, concat=False)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
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


