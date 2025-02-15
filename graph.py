import networkx as nx
import numpy as np

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
            'risk_level': 'high' if obj['objectType'] in ['Knife', 'StoveKnob'] else 'low',
            'state':'default'
        })
    print(object_info)
    return object_info

