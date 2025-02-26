import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv

class PredictiveGraphormer(nn.Module):
    def __init__(self, in_feats, hidden_dim, num_classes, num_heads, time_window=5):
        super(PredictiveGraphormer, self).__init__()
        
        self.num_heads = num_heads
        self.hidden_dim  = hidden_dim

        self.node_emb = nn.Linear(in_feats, hidden_dim)
        self.edge_emb = nn.Linear(6, hidden_dim)  # 6 维边特征
        self.transformer = TransformerConv(hidden_dim, hidden_dim // num_heads, heads=num_heads)

        # Transformer 处理时间信息
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # 预测下个时刻的危险度
        self.fc_out = nn.Linear(hidden_dim, num_classes)

        self.time_window = time_window

    def forward(self, x_seq, edge_index_seq, edge_attr_seq):
        """
        x_seq: [batch, time_window, num_nodes, in_feats]
        edge_index_seq: [batch, time_window, 2, num_edges]
        edge_attr_seq: [batch, time_window, num_edges, 6]
        """

        batch_size, time_window, num_nodes, in_feats = x_seq.shape

        # 处理所有时间窗口的节点特征和边特征
        x_seq = self.node_emb(x_seq.view(-1, in_feats)).view(batch_size, time_window, num_nodes, -1)
        edge_attr_seq = self.edge_emb(edge_attr_seq.view(-1, 6)).view(batch_size, time_window, -1, -1)

        # 通过 Transformer 处理每一帧
        for t in range(time_window):
            x_seq[:, t] = self.transformer(x_seq[:, t], edge_index_seq[:, t], edge_attr_seq[:, t])

        # Transformer 处理时间序列信息
        temporal_out = self.temporal_transformer(x_seq.permute(1, 0, 2, 3).contiguous().view(time_window, batch_size, -1))

        # 预测下一步
        out = self.fc_out(temporal_out[-1])  # 取最后一个时间步

        return out

class Graphormer(nn.Module):
    def __init__(self, in_feats, hidden_dim, num_classes, num_heads,edge_feat, dropout=0.3):
        super(Graphormer, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # 节点嵌入
        self.node_emb = nn.Linear(in_feats, hidden_dim)
        
        # 边嵌入，确保输出维度可以被 num_heads 整除
        self.edge_emb = nn.Linear(edge_feat, hidden_dim)

        # Transformer 层
        self.transformer = TransformerConv(hidden_dim, hidden_dim // num_heads, heads=num_heads)
        
        self.dropout = nn.Dropout(dropout)  # 添加 dropout

        # 修改最终分类层，让它在 **边** 级别上进行预测
        self.edge_fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, edge_attr):
        """
        x: 节点特征 [num_nodes, in_feats]
        edge_index: 边索引 [2, num_edges]
        edge_attr: 边特征 [num_edges, 2]
        """
        num_edges = edge_attr.shape[0]  # 获取 num_edges

        # 处理节点特征
        x = self.node_emb(x)

        # 处理边特征
        edge_attr = self.edge_emb(edge_attr)  # 变成 [num_edges, hidden_dim]
        edge_attr = edge_attr.view(num_edges, self.num_heads, self.hidden_dim // self.num_heads)

        # Transformer 计算（返回的仍然是节点特征）
        x = self.transformer(x, edge_index, edge_attr)
        
        x = self.dropout(x)  # 让模型不要死记硬背

        # **选取边对应的节点特征**
        edge_out = x[edge_index[0]] + x[edge_index[1]]  # 取边两端节点特征的和

        # **通过 edge_fc 预测边的类别**
        edge_preds = self.edge_fc(edge_out)  # 输出 (num_edges, num_classes)

        return edge_preds  # 直接输出边的预测结果

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")

    def forward(self, preds, targets):
        ce_loss = self.ce_loss(preds, targets)
        p_t = torch.exp(-ce_loss)  # 计算正确概率
        focal_weight = self.alpha * (1 - p_t) ** self.gamma  # 计算 Focal Loss 权重
        return (focal_weight * ce_loss).mean()  # 计算最终损失

# Run a simple test
if __name__ == "__main__":
    num_nodes = 10
    num_edges = 20
    in_feats = 5
    hidden_dim = 64
    num_classes = 2
    num_heads = 4

    # Fake data for testing
    node_feats = torch.rand((num_nodes, in_feats))  # Random node features
    edge_index = torch.randint(0, num_nodes, (2, num_edges))  # Random edges
    edge_attr = torch.rand((num_edges, 2))  # Random edge features attr dim=2

    # Initialize model
    model = Graphormer(in_feats, hidden_dim, num_classes, num_heads, edge_feat=2)

    # Forward pass
    output = model(node_feats, edge_index, edge_attr)

    print("✅ Graphormer Output Shape:", output.shape)  # Should be [num_nodes, num_classes]