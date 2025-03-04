import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import threading
import numpy as np
import networkx as nx
from collections import defaultdict
from ai2thor.controller import Controller
from scipy.spatial import distance
from typing import Tuple
from collections import deque
import random
from utils.task_agent import generate_task_sequence,parse_task_sequence
from utils.graph_utils import *
from utils.control_policy import *

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
    with open("./experiments/video/video_tasks.json") as f:
        aq = json.load(f)['task_sequence']
    cp.add_action_list(aq)
    print("starting executing!")
    # execute plan
    task_execution_thread = threading.Thread(target=cp.task_execution_loop)
    task_execution_thread.start()

if __name__ == "__main__":
    main()
