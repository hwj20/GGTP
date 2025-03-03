from torch_geometric.data import Data, Batch, Dataset
from collections import defaultdict
import networkx as nx
import numpy as np
import json
from utils.control_policy import distance_pts
import torch
from model.graphormer import *


# get environment data from AI2-THOR
def get_environment_data(controller, get_value = True):
    event = controller.step(action="Pass")
    object_info = []
    for obj in event.metadata['objects']:
        object_info.append({
            'name': obj['name'],
            'type': obj['objectType'],
            'pos': obj['position'] if get_value else obj['position'].values(),
            'state':'default'
        })
    return object_info, [obj['name'] for obj in object_info]

# Process AI2-THOR scene data into GNN-compatible format
def process_graph(data, get_label =True):
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
        edge_index.append([v, u])  # Convert to an undirected graph
        edge_index.append([u, v])
        v_map = {'None':0, 'high':1.0, 'medium':0.5, 'low':0.25}
        edge_feats.append([edge["distance"], 2*v_map[edge['risk_level']]])
        edge_feats.append([edge["distance"], 2*v_map[edge['risk_level']]])
        if get_label:
            edge_labels.append(edge["edge_type"])
            edge_labels.append(edge["edge_type"])

    edge_feats = torch.tensor(edge_feats, dtype=torch.float32)
    edge_labels = torch.tensor(edge_labels, dtype=torch.long)
    edge_index = torch.tensor(edge_index, dtype=torch.long).T

    if get_label:
        return Data(x=node_feats, edge_index=edge_index, edge_attr=edge_feats, y=edge_labels)
    else:
        return Data(x=node_feats, edge_index=edge_index, edge_attr=edge_feats)

# Oversample positive samples in the dataset
# Note: we use oversample with factor=0(no oversampling) in training dataset because
# __getitem__() in this function doesn't need to call processing graph for ecah call, 
# which is faster in training. As for the val and test datasets, just let it go :) 
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


def build_environment_graph(objects):
    """
    Construct environment data as two lists: `nodes` and `edges`.

    Parameters:
    - objects (list): A list of object dictionaries, each containing:
        - "name": Object identifier
        - "type": Object type (e.g., Knife, Table)
        - "pos": Object position (x, y, z)
        - "state": Object state (e.g., Open, Closed)
    
    Returns:
    - nodes (list): List of object nodes with features.
    - edges (list): List of relationships between objects with risk info.
    """
    with open("./experiments/data/danger_info.json") as f:
        dangers = json.load(f)
    danger_map = defaultdict(dict)

    for i in range(len(dangers)):
        danger_info = dangers[i]
        danger_map[danger_info['type1']][danger_info['type2']] = i
        danger_map[danger_info['type2']][danger_info['type1']] = i

    nodes = []
    edges = []


    # Add nodes
    for idx, obj in enumerate(objects):
        nodes.append({
            "name":obj['name'],
            "node_id": idx,
            "node_type": obj["type"],
            "features": {
                "temperature": obj.get("temperature", 20),
                "energy_source": obj.get("energy_source", "none"),
                "position": obj["pos"].values()
            }
        })

    # Add edges
    for i, obj1 in enumerate(objects):
        for j, obj2 in enumerate(objects):
            if i < j:
                dist = distance_pts(obj1["pos"], obj2["pos"])

                # Default risk values
                risk_level = "None"
                risk_type = []
                attention_bias = 0
                # Check if the object pair exists in danger_map
                if obj1["type"] in danger_map and obj2["type"] in danger_map[obj1["type"]]:
                    # print(obj1,obj2)
                    risk_idx = danger_map[obj1["type"]][obj2["type"]]
                    risk_info = dangers[risk_idx]

                    risk_level = risk_info["danger_level"]
                    risk_type = risk_info["risk_type"]
                    attention_bias = max(1 / dist if dist > 0 else 1.0,3)  # Distance-based risk weight

                edges.append({
                    "edge_id": len(edges),
                    "node1_id": i,
                    "node2_id": j,
                    "distance": round(dist, 2),
                    "risk_level": risk_level,
                    "risk_type": risk_type,
                    "attention_bias": attention_bias
                })

    return nodes, edges

def receive_safety_notice(nodes, edges):
    """
    Uses the trained Graphormer model to predict potential hazards in the environment.
    - Processes AI2-THOR scene data into GNN-compatible format.
    - Loads and applies the trained Graphormer model.
    - Returns a safety warning message.

    Generate a natural language description for detected hazardous edges.
    """
    
    model_path = "./model/graphormer_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Graphormer(in_feats=5, hidden_dim=64, num_classes=2, num_heads=4, edge_feat=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Process the graph data
    graph_data = process_graph({"nodes": nodes, "edges": edges}, False)
    
    # Move data to the appropriate device
    graph_data = graph_data.to(device)

    # Model inference
    edge_preds = model(graph_data.x, graph_data.edge_index, graph_data.edge_attr)
    probs = torch.softmax(edge_preds, dim=1)[:, 1]  # Probability of being hazardous
    hazard_edges = (probs > 0.21).nonzero(as_tuple=True)[0]  # Identify hazardous edges

    if len(hazard_edges) == 0:
        return "No hazardous situations detected."

    # Convert indices to object names
    id_to_name = {node["node_id"]: node["name"] for node in nodes}  # Map node IDs to object names
    
    hazard_descriptions = []
    added = set()
    for edge_idx in hazard_edges:
        u, v = graph_data.edge_index[:, edge_idx].cpu().numpy()
        obj1, obj2 = id_to_name.get(u, f"Object_{u}"), id_to_name.get(v, f"Object_{v}")
        # filter repeated advice
        if (obj2, obj1) in added or (obj1,obj2) in added:
            continue
        hazard_descriptions.append(f"Dangerous edge detected: {obj1} → {obj2}")
        added.add((obj1,obj2))

    return "\n".join(hazard_descriptions)

# A single LTL rule: Baby and Knife cannot show up in one scene both
def receive_safety_notice_ltl(scene_objects ):
    if any("Knife" in obj for obj in scene_objects) and any("Baby" in obj for obj in scene_objects):
        return "DANGER: Baby too close to knife!"
    return "Safe"