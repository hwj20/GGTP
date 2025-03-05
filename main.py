# A demo of the pipline

import numpy as np
import networkx as nx
from collections import defaultdict, deque
from ai2thor.controller import Controller
from scipy.spatial import distance
from typing import Tuple
import random
import os
from utils.task_agent import generate_task_sequence, parse_task_sequence
from utils.graph_utils import *
from utils.control_policy import *

# Initialize the environment
floor_no = 1
c = Controller(height=1000, width=1000)
c.reset(f"FloorPlan{floor_no}")
cp = ControlPolicy(c, "tag")

# Define robot capabilities
robot_activities = ["GoToObject", "PickupObject", "PutObject", "SwitchOn", "SwitchOff", "SliceObject"]
robots = [
    {"name": "robot1", "skills": robot_activities},
]
cp.init_robots(robots)

print("Robot initialized!")

def main():
    enable_safety_aware = False
    method = "graphomer"
    
    # 1. Retrieve object information
    env_objects, obj_lists = get_environment_data(controller=c)
    
    safety_notice = 'Safe'
    if enable_safety_aware:
        if method == 'graphomer':
            # 2. Construct the environment graph
            nodes, edges = build_environment_graph(env_objects)
            
            # 3. Receive safety notifications
            safety_notice = receive_safety_notice(nodes, edges)
        
        elif method == 'ltl':
            # Note: please modify the safety rules
            # 3. Receive safety notifications
            safety_notice = receive_safety_notice_ltl(obj_lists)
    
    # 4. Generate task sequence 
    task_description = "pick up an apple and put it on any container"
    task_sequence_json = generate_task_sequence(task_description, robot_activities, obj_lists, safety_notice)
    
    # 5. Parse task sequence
    action_queue = parse_task_sequence(task_sequence_json)
    
    print(action_queue)
    cp.add_action_list(action_queue)
    
    print("Starting execution!")

    # 6. Execute tasks
    cp.run_task_thread()

if __name__ == "__main__":
    main()

