import openai
import math
import re
import subprocess
import time
import threading
import numpy as np
import networkx as nx
from collections import defaultdict, deque
from ai2thor.controller import Controller
from scipy.spatial import distance
from typing import Tuple
import random
import os
from utils.task_agent import generate_task_sequence, parse_task_sequence
from graph import *
from control_policy import *

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
    method = "graphomer"
    start_time = time.time()
    
    # 1. **Retrieve object information**
    step_start = time.time()
    env_objects, obj_lists = get_environment_data(controller=c)
    step_end = time.time()
    print(f"Step 1: Retrieving object information took {step_end - step_start:.4f} seconds")
    
    if method == 'graphomer':
        # 2. **Construct the environment graph**
        step_start = time.time()
        nodes, edges = build_environment_graph(env_objects)
        step_end = time.time()
        print(f"Step 2: Constructing the environment graph took {step_end - step_start:.4f} seconds")
        
        # 3. **Receive safety notifications**
        step_start = time.time()
        safety_notice = receive_safety_notice(nodes, edges)
        step_end = time.time()
        print(f"Step 3: Receiving safety notifications took {step_end - step_start:.4f} seconds")
    
    elif method == 'ltl':
        step_start = time.time()
        safety_notice = receive_safety_notice_ltl(obj_lists)
        step_end = time.time()
        print(f"Step 3: Receiving safety notifications took {step_end - step_start:.4f} seconds")
    
    # 4. **Generate task sequence (using Graphormer safety perception)**
    step_start = time.time()
    task_description = "goto refrigerator"
    task_sequence_json = generate_task_sequence(task_description, robot_activities, obj_lists, safety_notice)
    step_end = time.time()
    print(f"Step 4: Generating task sequence took {step_end - step_start:.4f} seconds")
    
    # 5. **Parse task sequence**
    global action_queue
    step_start = time.time()
    action_queue = parse_task_sequence(task_sequence_json)
    step_end = time.time()
    print(f"Step 5: Parsing task sequence took {step_end - step_start:.4f} seconds")
    
    print(action_queue)
    cp.add_action_list(action_queue)
    
    print("Starting execution!")

    # 6. **Execute tasks**
    # step_start = time.time()
    # task_execution_thread = threading.Thread(target=cp.task_execution_loop)
    # task_execution_thread.start()
    # step_end = time.time()
    # print(f"Step 6: Starting task execution thread took {step_end - step_start:.4f} seconds")
    
    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.4f} seconds")

if __name__ == "__main__":
    main()
