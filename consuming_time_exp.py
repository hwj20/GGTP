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
from task_agent import generate_task_sequence, parse_task_sequence
from graph import *
from control_policy import *

floor_no = 1
c = Controller(height=1000, width=1000)
c.reset(f"FloorPlan{floor_no}")
cp = ControlPolicy(c, "tag")
robot_activities = ["GoToObject", "PickupObject", "PutObject", "SwitchOn", "SwitchOff", "SliceObject"]
robots = [
    {"name": "robot1", "skills": robot_activities},
]
cp.init_robots(robots)

print("Robot initialized!")

def main():
    start_time = time.time()
    
    # 1. **获取物体信息**
    step_start = time.time()
    env_objects, obj_lists = get_environment_data(controller=c)
    step_end = time.time()
    print(f"Step 1: 获取物体信息耗时 {step_end - step_start:.4f} 秒")
    
    # print(obj_lists)
    
    # 2. **构建环境图**
    # step_start = time.time()
    # nodes, edges = build_environment_graph(env_objects)
    # step_end = time.time()
    # print(f"Step 2: 构建环境图耗时 {step_end - step_start:.4f} 秒")
    
    # # 3. **接收安全通知**
    step_start = time.time()
    safety_notice = receive_safety_notice_ltl(obj_lists)
    # safety_notice = receive_safety_notice(nodes, edges)
    step_end = time.time()
    print(f"Step 3: 接收安全通知耗时 {step_end - step_start:.4f} 秒")
    
    # 4. **生成任务序列（使用 Graphormer 安全感知）**
    step_start = time.time()
    task_description = "goto refrigerator"
    task_sequence_json = generate_task_sequence(task_description, robot_activities, obj_lists, safety_notice)
    step_end = time.time()
    print(f"Step 4: 生成任务序列耗时 {step_end - step_start:.4f} 秒")
    
    # print(f"Generated Task Sequence:\n{task_sequence_json}")
    
    global action_queue
    step_start = time.time()
    action_queue = parse_task_sequence(task_sequence_json)
    step_end = time.time()
    print(f"Step 5: 解析任务序列耗时 {step_end - step_start:.4f} 秒")
    
    print(action_queue)
    cp.add_action_list(action_queue)
    
    print("Starting execution!")
    
    # 5. **执行任务**
    # step_start = time.time()
    # task_execution_thread = threading.Thread(target=cp.task_execution_loop)
    # task_execution_thread.start()
    # step_end = time.time()
    # print(f"Step 6: 启动任务执行线程耗时 {step_end - step_start:.4f} 秒")
    
    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.4f} 秒")

if __name__ == "__main__":
    main()
