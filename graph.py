from collections import defaultdict
import networkx as nx
import numpy as np
import json
from control_policy import distance_pts

# 构建厨房环境的图
def build_kitchen_graph():
    G = nx.Graph()

    nodes = [
        {"name": "knife", "type": "object", "risk": "cut", "location": [0.2, 0.8]},
        {"name": "stove", "type": "object", "risk": "burn", "state": "on", "location": [0.5, 1.2]},
        {"name": "cutting_board", "type": "object", "risk": "none", "location": [0.3, 0.6]},
        {"name": "child", "type": "human", "risk": "high", "state": "running", "location": [0.8, 1.5]},
        {"name": "robot", "type": "agent", "risk": "none", "location": [1.0, 1.0]}
    ]

    for node in nodes:
        G.add_node(node["name"], type=node["type"], risk=node["risk"], state=node.get("state", "none"), location=node["location"])

    # 添加边，模拟物体之间的交互
    edges = [
        ("knife", "cutting_board"),  # 刀和砧板关联
        ("stove", "cutting_board"),  # 炉灶和砧板关联
        ("child", "knife"),  # 孩子和刀
        ("child", "stove"),  # 孩子和炉灶
        ("robot", "knife")  # 机器人和刀
    ]
    
    for edge in edges:
        G.add_edge(*edge)

    return G

def get_environment_data(controller):
    event = controller.step(action="Pass")
    object_info = []
    for obj in event.metadata['objects']:
        object_info.append({
            'name': obj['name'],
            'type': obj['objectType'],
            'pos': obj['position'],
            # 'risk_level': 'high' if obj['objectType'] in ['Knife', 'StoveKnob'] else 'low',
            'state':'default'
        })
    # print(object_info)
    return object_info, [obj['name'] for obj in object_info]


def build_environment_graph(objects):
    """
    Construct environment data as two lists: `nodes` and `edges`.

    Parameters:
    - objects (list): A list of object dictionaries, each containing:
        - "name": Object identifier
        - "type": Object type (e.g., Knife, Table)
        - "pos": Object position (x, y, z)
        - "state": Object state (e.g., Open, Closed)
        - "risk_level": Risk level (e.g., Low, Medium, High)
    
    Returns:
    - nodes (list): List of object nodes with features.
    - edges (list): List of relationships between objects with risk info.
    """
    with open("./data/danger_info.json") as f:
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
            "node_id": idx,
            "node_type": obj["type"],
            "features": {
                "temperature": obj.get("temperature", 20),
                "energy_source": obj.get("energy_source", "none"),
                "position": obj["pos"]
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
                    risk_idx = danger_map[obj1["type"]][obj2["type"]]
                    risk_info = dangers[risk_idx]

                    risk_level = risk_info["danger_level"]
                    risk_type = risk_info["risk_type"]
                    attention_bias = 1 / dist if dist > 0 else 1.0  # Distance-based risk weight

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

def receive_safety_notice():
    return ""