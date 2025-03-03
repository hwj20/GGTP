import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv

# Explored but not enabled
class PredictiveGraphormer(nn.Module):
    def __init__(self, in_feats, hidden_dim, num_classes, num_heads, time_window=5):
        super(PredictiveGraphormer, self).__init__()
        
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        # Node and edge embeddings
        self.node_emb = nn.Linear(in_feats, hidden_dim)
        self.edge_emb = nn.Linear(6, hidden_dim)  # Edge features with dimension 6

        # Transformer for spatial relationships
        self.transformer = TransformerConv(hidden_dim, hidden_dim // num_heads, heads=num_heads)

        # Transformer to process temporal information
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # Predict hazard level at the next timestep
        self.fc_out = nn.Linear(hidden_dim, num_classes)

        self.time_window = time_window

    def forward(self, x_seq, edge_index_seq, edge_attr_seq):
        """
        x_seq: [batch, time_window, num_nodes, in_feats]
        edge_index_seq: [batch, time_window, 2, num_edges]
        edge_attr_seq: [batch, time_window, num_edges, 6]
        """

        batch_size, time_window, num_nodes, in_feats = x_seq.shape

        # Process node and edge features for all time steps
        x_seq = self.node_emb(x_seq.view(-1, in_feats)).view(batch_size, time_window, num_nodes, -1)
        edge_attr_seq = self.edge_emb(edge_attr_seq.view(-1, 6)).view(batch_size, time_window, -1, -1)

        # Apply Transformer for each time frame
        for t in range(time_window):
            x_seq[:, t] = self.transformer(x_seq[:, t], edge_index_seq[:, t], edge_attr_seq[:, t])

        # Use Transformer to process temporal sequences
        temporal_out = self.temporal_transformer(x_seq.permute(1, 0, 2, 3).contiguous().view(time_window, batch_size, -1))

        # Predict the next step
        out = self.fc_out(temporal_out[-1])  # Take the last time step

        return out

class Graphormer(nn.Module):
    def __init__(self, in_feats, hidden_dim, num_classes, num_heads, edge_feat, dropout=0.3):
        super(Graphormer, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Node embedding
        self.node_emb = nn.Linear(in_feats, hidden_dim)
        
        # Edge embedding, ensuring the output dimension is divisible by num_heads
        self.edge_emb = nn.Linear(edge_feat, hidden_dim)

        # Transformer layer
        self.transformer = TransformerConv(hidden_dim, hidden_dim // num_heads, heads=num_heads)
        
        self.dropout = nn.Dropout(dropout)  # Add dropout for regularization

        # Final classification layer for edge-based prediction
        self.edge_fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, edge_attr):
        """
        x: Node features [num_nodes, in_feats]
        edge_index: Edge indices [2, num_edges]
        edge_attr: Edge features [num_edges, 2]
        """
        num_edges = edge_attr.shape[0]  # Get number of edges

        # Process node features
        x = self.node_emb(x)

        # Process edge features
        edge_attr = self.edge_emb(edge_attr)  # Shape: [num_edges, hidden_dim]
        edge_attr = edge_attr.view(num_edges, self.num_heads, self.hidden_dim // self.num_heads)

        # Compute Transformer (output is still node features)
        x = self.transformer(x, edge_index, edge_attr)
        
        x = self.dropout(x)  # Apply dropout to prevent overfitting

        # Select node features for edges
        edge_out = x[edge_index[0]] + x[edge_index[1]]  # Sum features from both edge nodes

        # Predict edge classification
        edge_preds = self.edge_fc(edge_out)  # Output shape: (num_edges, num_classes)

        return edge_preds  # Return edge-level predictions

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")

    def forward(self, preds, targets):
        ce_loss = self.ce_loss(preds, targets)
        p_t = torch.exp(-ce_loss)  # Compute probability of the correct class
        focal_weight = self.alpha * (1 - p_t) ** self.gamma  # Compute Focal Loss weight
        return (focal_weight * ce_loss).mean()  # Compute final loss

# Run a simple test
if __name__ == "__main__":
    num_nodes = 10
    num_edges = 20
    in_feats = 5
    hidden_dim = 64
    num_classes = 2
    num_heads = 4

    # Generate fake test data
    node_feats = torch.rand((num_nodes, in_feats))  # Random node features
    edge_index = torch.randint(0, num_nodes, (2, num_edges))  # Random edge indices
    edge_attr = torch.rand((num_edges, 2))  # Random edge features with dim=2

    # Initialize model
    model = Graphormer(in_feats, hidden_dim, num_classes, num_heads, edge_feat=2)

    # Forward pass
    output = model(node_feats, edge_index, edge_attr)

    print("✅ Graphormer Output Shape:", output.shape)  # Expected: [num_edges, num_classes]
