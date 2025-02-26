from dangerous_types import dangerous_types
import torch.nn.functional as F
import collections
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, Batch, Dataset
from torch.utils.data import DataLoader
from model.graphormer import Graphormer, FocalLoss
import random
import numpy as np
import matplotlib.pyplot as plt



# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# def ltl_baseline_evaluate(loader):
#     """ Simulates an LTL-based model that only considers object types, ignoring spatial awareness. """
#     total_1 = 0  # True positive count
#     correct_1 = 0  # Correctly classified danger edges
#     predicted_1 = 0  # Total predicted danger edges

#     with torch.no_grad():
#         for batch in loader:
#             batch = batch.to(device)
            
#             # LTL-based rule: classify based on "object type" instead of actual risk
#             ltl_preds = torch.zeros_like(batch.y, device=device)  # Assume all edges are safe
            
#             for i, (src, dst) in enumerate(batch.edge_index.T):
#                 src_type = batch.x[src][-1].item()  # find it in Edge index
#                 dst_type = batch.x[dst][-1].item()
                
#                 if src_type in dangerous_types or dst_type in dangerous_types:
#                     ltl_preds[i] = 1  # Mark as risky, ignoring actual spatial position

#             total_1 += (batch.y == 1).sum().item()
#             correct_1 += ((ltl_preds == 1) & (batch.y == 1)).sum().item()
#             predicted_1 += (ltl_preds == 1).sum().item()

#     recall = correct_1 / total_1 if total_1 > 0 else 0
#     precision = correct_1 / predicted_1 if predicted_1 > 0 else 0

#     print(f"LTL Baseline -> Recall: {recall:.4f}, Precision: {precision:.4f}")
#     return recall, precision

# Load dataset
with open("./data/graph_dataset.json", "r") as f:
    dataset = json.load(f)

# Process AI2-THOR scene data into GNN-compatible format
def process_graph(data):
    nodes = data["nodes"]
    edges = data["edges"]

    def get_features(node):
        temp_map = {"cold": 5.0, "roomtemp": 20.0, "hot": 50.0}
        temperature = temp_map.get(node["features"].get("temperature", "roomtemp"), 20.0)
        
        return [
            temperature,
            1.0 if node["features"].get("energy_source", "none") != "none" else 0.0,
            *node["features"].get("position", [0.0, 0.0, 0.0])
        ]

    # nodes_type = {node['node_id']:node['node_type'] for node in nodes}
    node_feats = torch.tensor([get_features(node) for node in nodes], dtype=torch.float32)

    edge_feats, edge_labels, edge_index = [], [], []
    for edge in edges:
        u, v = edge["node1_id"], edge["node2_id"]
        edge_index.append([u, v])
        edge_index.append([v, u])  # Convert to an undirected graph
        v_map = {'None':0, 'high':1.0, 'medium':0.5, 'low':0.25}
        edge_feats.append([edge["distance"], 2*v_map[edge['risk_level']]])
        edge_feats.append([edge["distance"], 2*v_map[edge['risk_level']]])
        edge_labels.append(edge["edge_type"])
        edge_labels.append(edge["edge_type"])

    edge_feats = torch.tensor(edge_feats, dtype=torch.float32)
    edge_labels = torch.tensor(edge_labels, dtype=torch.long)
    edge_index = torch.tensor(edge_index, dtype=torch.long).T

    return Data(x=node_feats, edge_index=edge_index, edge_attr=edge_feats, y=edge_labels)

# Oversample positive samples in the dataset
class OversampleGraphDataset(Dataset):
    def __init__(self, data_list, oversample_factor=5):
        self.graphs = []

        for data in data_list:
            g = process_graph(data)  # Process each graph
            mask = g.y == 1  # Identify all positive class edges

            if mask.sum() > 0:  # Ensure at least one positive edge exists
                extra_edges = []
                extra_edge_attrs = []
                extra_labels = []

                for _ in range(oversample_factor):  # Duplicate samples
                    extra_edges.append(g.edge_index[:, mask].clone())  
                    extra_edge_attrs.append(g.edge_attr[mask].clone())  
                    extra_labels.append(g.y[mask].clone())  

                # Merge with the original graph
                g.edge_index = torch.cat([g.edge_index, *extra_edges], dim=1)
                g.edge_attr = torch.cat([g.edge_attr, *extra_edge_attrs], dim=0)
                g.y = torch.cat([g.y, *extra_labels], dim=0)

            self.graphs.append(g)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]

# Standard dataset class without oversampling
class GraphDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return process_graph(self.data_list[idx])

# Load dataset
train_data = OversampleGraphDataset(dataset["train"], 0)
val_data = GraphDataset(dataset["val"])
test_data = GraphDataset(dataset["test"])

# Define collate functions
def collate_fn(batch):
    return Batch.from_data_list(batch)

def scene_collate_fn(batch):
    return Batch.from_data_list(batch)

# Create DataLoaders
train_loader = DataLoader(train_data, batch_size=8, shuffle=True, collate_fn=scene_collate_fn)
val_loader = DataLoader(val_data, batch_size=8, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_data, batch_size=8, shuffle=False, collate_fn=collate_fn)

# Initialize the Graphormer model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = Graphormer(in_feats=5, hidden_dim=64, num_classes=2, num_heads=4, edge_feat=2).to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
criterion = FocalLoss(alpha=0.5, gamma=1.0)

# Train function
def train():
    recall_list = []
    precision_list = []
    epochs = []
    
    for epoch in range(1, 600):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            preds = model(batch.x, batch.edge_index, batch.edge_attr)
            loss = criterion(preds, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            recall, precision = evaluate(val_loader)
            recall_list.append(recall)
            precision_list.append(precision)
            epochs.append(epoch)
            print(f"Epoch {epoch}: Loss = {total_loss / len(train_loader):.4f}")

# Evaluation function
def evaluate(loader, threshold=0.1):
    model.eval()
    total_1 = sum_1 = correct_1 = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            probs = F.softmax(model(batch.x, batch.edge_index, batch.edge_attr), dim=1)[:, 1]
            preds = (probs > threshold).long()
            total_1 += (batch.y == 1).sum().item()
            correct_1 += ((preds == 1) & (batch.y == 1)).sum().item()
            sum_1 += (preds == 1).sum().item()
    recall = correct_1 / total_1 if total_1 > 0 else 0
    precision = correct_1 / sum_1 if sum_1 > 0 else 0
    print(f"Threshold = {threshold:.2f} -> Recall: {recall:.4f}, Precision: {precision:.4f}")
    return recall, precision

# Function to evaluate random predictions
def evaluate_random(loader, threshold=0.1):
    total_1 = 0  # True positive count
    correct_1 = 0  # Correctly predicted positive count
    predicted_1 = 0  # Predicted positive count
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            random_prob = torch.rand(batch.y.shape, device=device) 
            random_preds = (random_prob > threshold).long()
            
            total_1 += (batch.y == 1).sum().item()
            correct_1 += ((random_preds == 1) & (batch.y == 1)).sum().item()
            predicted_1 += (random_preds == 1).sum().item()
    
    recall = correct_1 / total_1 if total_1 > 0 else 0
    precision = correct_1 / predicted_1 if predicted_1 > 0 else 0
    
    return recall, precision

train()
# def plot_threshold_evaluation():
#     thresholds = np.linspace(0.0, 1.0, 20)
#     model_recalls, model_precisions = [], []
#     random_recalls, random_precisions = [], []
    
#     for threshold in thresholds:
#         recall, precision = evaluate(test_loader, threshold=threshold)
#         model_recalls.append(recall)
#         model_precisions.append(precision)
        
#         random_recall, random_precision = evaluate_random(test_loader, threshold=threshold)  # Evaluate random predictions
#         random_recalls.append(random_recall)
#         random_precisions.append(random_precision)
    
#     plt.figure(figsize=(10, 6))
#     plt.plot(thresholds, model_recalls, label="Model Recall", linestyle='-', linewidth=2, marker='o')
#     plt.plot(thresholds, model_precisions, label="Model Precision", linestyle='--', linewidth=2, marker='s')
#     plt.plot(thresholds, random_recalls, label="Random Guess Recall", linestyle='-', linewidth=2, marker='o', color='gray', alpha=0.6)
#     plt.plot(thresholds, random_precisions, label="Random Guess Precision", linestyle='--', linewidth=2, marker='s', color='black', alpha=0.6)

#     plt.xlabel("Threshold")
#     plt.ylabel("Score")
#     plt.title("Model vs. Random Guess: Recall and Precision")
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# plot_threshold_evaluation()



# # Run baseline evaluation on validation and test sets
# ltl_recall_val, ltl_precision_val = ltl_baseline_evaluate(val_loader)
# ltl_recall_test, ltl_precision_test = ltl_baseline_evaluate(test_loader)

# # Plot comparison against Graphormer
# def plot_baseline_comparison():
#     models = ["Graphormer", "LTL Baseline"]
#     recall_scores = [evaluate(val_loader, threshold=0.1)[0], ltl_recall_val]
#     precision_scores = [evaluate(val_loader, threshold=0.1)[1], ltl_precision_val]
    
#     x = np.arange(len(models))
#     width = 0.3
    
#     plt.figure(figsize=(8, 5))
#     plt.bar(x - width/2, recall_scores, width, label="Recall")
#     plt.bar(x + width/2, precision_scores, width, label="Precision")
    
#     plt.xlabel("Models")
#     plt.ylabel("Score")
#     plt.title("Graphormer vs. LTL Baseline (Validation Set)")
#     plt.xticks(x, models)
#     plt.legend()
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
#     plt.show()

# plot_baseline_comparison()

