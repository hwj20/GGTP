import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import openai
import math
import re
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
from task_agent import generate_task_sequence,parse_task_sequence
from graph import *
from control_policy import *

floor_no = 2
c = Controller(height=1000, width=1000)
c.reset(f"FloorPlan{floor_no}")
cp = ControlPolicy(c,'tag')
robot_activities =  ["GoToObject", "PickupObject", "PutObject", "SwitchOn", "SwitchOff", "SliceObject"]
robots = [
    {"name": "robot1", "skills":robot_activities},
]
cp.init_robots(robots)

print("Robot initialized!")

def main():
    
    # 1. **获取物体信息**
    # env_objects, obj_lists = get_environment_data(controller=c)
    # print(obj_lists)
    # input()

    # 2. **构建环境图**
    # nodes,edges = build_environment_graph(env_objects)

    # safety_notice = "No advice"
    # safety_notice = receive_safety_notice(nodes,edges)
    # # 3. **生成任务序列（使用 Graphormer 安全感知）**
    # task_description = "goto refrigerator"
    # task_sequence_json = generate_task_sequence(task_description,robot_activities, obj_lists,safety_notice)
    # print(f"Generated Task Sequence:\n{task_sequence_json}")
    # global action_queue
    # action_queue =  parse_task_sequence(task_sequence_json)
    # print(action_queue)
    # cp.add_action_list(action_queue)

    with open("./video/video_tasks.json") as f:
        aq = json.load(f)['task_sequence']
    cp.add_action_list(aq)
    print("starting executing!")
    # 4. **执行任务**
    task_execution_thread = threading.Thread(target=cp.task_execution_loop)
    task_execution_thread.start()

if __name__ == "__main__":
    main()
