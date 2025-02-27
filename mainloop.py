import json
import openai
import math
import re
import shutil
import subprocess
import time
import threading
import numpy as np
import networkx as nx
from collections import defaultdict
from ai2thor.controller import Controller
from scipy.spatial import distance
from typing import Tuple
from collections import deque
import random
import os
from task_agent import generate_task_sequence,parse_task_sequence
from graph import *
from control_policy import *

floor_no = 1
c = Controller(height=1000, width=1000)
c.reset(f"FloorPlan{floor_no}")
cp = ControlPolicy(c)
robot_activities =  ["GoToObject", "PickupObject", "PutObject", "SwitchOn", "SwitchOff", "SliceObject"]
robots = [
    {"name": "robot1", "skills":robot_activities},
]
cp.init_robots(robots)

print("Robot initialized!")

with open("./data/danger_info.json") as f:
    dangers = json.load(f)
danger_map = defaultdict(dict)

for i in range(len(dangers)):
    danger_info = dangers[i]
    danger_map[danger_info['type1']][danger_info['type2']] = i
    danger_map[danger_info['type2']][danger_info['type1']] = i

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


def main():
    # 1. **获取物体信息**
    env_objects, obj_lists = get_environment_data(controller=c)
    print(obj_lists)
    # input()

    # 2. **构建环境图**
    nodes,edges = build_environment_graph(env_objects)

    # 3. **生成任务序列（使用 Graphormer 安全感知）**
    task_description = "goto refrigerator"
    task_sequence_json = generate_task_sequence(task_description,robot_activities, obj_lists)
    print(f"Generated Task Sequence:\n{task_sequence_json}")
    global action_queue
    action_queue =  parse_task_sequence(task_sequence_json)
    print(action_queue)
    cp.add_action_list(action_queue)

    print("starting executing!")
    # 4. **执行任务**
    task_execution_thread = threading.Thread(target=cp.task_execution_loop)
    task_execution_thread.start()

if __name__ == "__main__":
    main()
