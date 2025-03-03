"""
Deprecated Model: Safe-GAT 

Initially, we considered using a Graph Attention Network (GAT) for safety-aware task planning. 
However, we later found that Graphormer provides a more **modern and structured** approach to modeling 
spatial and semantic relationships. As a result, this Safe-GAT implementation has been **deprecated** 
in favor of Graphormer.

This script remains here for reference but is no longer actively used.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

# Define the Safe-GAT model
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

# Convert a networkx graph to PyTorch Geometric data format
def graph_to_torch_data(G):
    node_list = list(G.nodes)
    node_features = torch.rand(len(node_list), 4)  # Randomly initialize node features
    edge_index = torch.tensor(
        [[node_list.index(e[0]), node_list.index(e[1])] for e in G.edges], 
        dtype=torch.long
    ).t().contiguous()
    
    return Data(x=node_features, edge_index=edge_index)

# Generate safety recommendations based on GNN output
def generate_safety_recommendations(GNN_output, graph):
    attention_nodes = []
    risk_relations = {}

    for i, (node, attr) in enumerate(graph.nodes(data=True)):
        attention_score = GNN_output[i][0].item()
        if attention_score > 0.5:  # Set attention threshold
            attention_nodes.append(node)
            for neighbor in graph.neighbors(node):
                risk_relations[f"{node} → {neighbor}"] = f"⚠️ {attr['risk']} related risk"

    return {
        "high_attention_nodes": attention_nodes,
        "risk_relations": risk_relations
    }
