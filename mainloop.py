import openai
import math
import re
import shutil
import subprocess
import time
import threading
import cv2
import numpy as np
import networkx as nx
from ai2thor.controller import Controller
from scipy.spatial import distance
from typing import Tuple
from collections import deque
import random
import os
from glob import glob
from task_agent import generate_task_sequence,parse_task_sequence
from graph import *

__file__ = 'testing'
# 初始化环境
floor_no = 1
c = Controller(height=1000, width=1000)
c.reset(f"FloorPlan{floor_no}")

# 定义机器人
robots = [
    {"name": "robot1", "skills": ["GoToObject", "PickupObject", "PutObject", "SwitchOn", "SwitchOff", "SliceObject"]},
]
no_robot = len(robots)

# initialize n agents into the scene
multi_agent_event = c.step(dict(action='Initialize', agentMode="default", snapGrid=False, gridSize=0.25, rotateStepDegrees=20, visibilityDistance=100, fieldOfView=90, agentCount=no_robot))

# add a top view camera
event = c.step(action="GetMapViewCameraProperties")
event = c.step(action="AddThirdPartyCamera", **event.metadata["actionReturn"])

# get reachabel positions
reachable_positions_ = c.step(action="GetReachablePositions").metadata["actionReturn"]
reachable_positions = positions_tuple = [(p["x"], p["y"], p["z"]) for p in reachable_positions_]

# initialize robots
for i in range(no_robot):
    init_pos = random.choice(reachable_positions_)
    c.step(dict(action="Teleport", position=init_pos, agentId=i))

print("机器人初始化完毕")

# 任务队列
action_queue = []
task_over = False


### **📍 空间建模（Graphormer 替代 GAT）**
def build_environment_graph(objects):
    """
    通过物体信息构建环境图
    """
    G = nx.Graph()
    for obj in objects:
        G.add_node(obj["name"], type=obj["type"], coordinates=obj["pos"], state=obj["state"], risk=obj['risk_level'])

    # 计算物体之间的空间关系
    for i, obj1 in enumerate(objects):
        for j, obj2 in enumerate(objects):
            if i < j:
                dist = distance_pts(obj1["pos"], obj2["pos"])
                G.add_edge(obj1["name"], obj2["name"], distance=dist)

    return G

def distance_pts(p1, p2):
    """
    计算两点间欧几里得距离
    """
    return ((p1['x'] - p2['x']) ** 2 + (p1['y'] - p2['y']) ** 2 +(p1['z'] - p2['z']) ** 2) ** 0.5

def closest_node(node, nodes, no_robot, clost_node_location):
    crps = []
    distances = distance.cdist([node], nodes)[0]
    dist_indices = np.argsort(np.array(distances))
    for i in range(no_robot):
        pos_index = dist_indices[(i * 5) + clost_node_location[i]]
        crps.append (nodes[pos_index])
    return crps

def GoToObject(robots, dest_obj):
    print ("Going to ", dest_obj)
    # check if robots is a list
    
    if not isinstance(robots, list):
        # convert robot to a list
        robots = [robots]
    no_agents = len (robots)
    # robots distance to the goal 
    dist_goals = [10.0] * len(robots)
    prev_dist_goals = [10.0] * len(robots)
    count_since_update = [0] * len(robots)
    clost_node_location = [0] * len(robots)
    
    # list of objects in the scene and their centers
    objs = list([obj["objectId"] for obj in c.last_event.metadata["objects"]])
    objs_center = list([obj["axisAlignedBoundingBox"]["center"] for obj in c.last_event.metadata["objects"]])

    # look for the location and id of the destination object
    for idx, obj in enumerate(objs):
        match = re.match(dest_obj, obj)
        if match is not None:
            dest_obj_id = obj
            dest_obj_center = objs_center[idx]
            break # find the first instance
        
    dest_obj_pos = [dest_obj_center['x'], dest_obj_center['y'], dest_obj_center['z']] 
    
    # closest reachable position for each robot
    # all robots cannot reach the same spot 
    # differt close points needs to be found for each robot
    crp = closest_node(dest_obj_pos, reachable_positions, no_agents, clost_node_location)
    
    goal_thresh = 0.3
    # at least one robot is far away from the goal
    
    while all(d > goal_thresh for d in dist_goals):
        for ia, robot in enumerate(robots):
            robot_name = robot['name']
            agent_id = int(robot_name[-1]) - 1
            
            # get the pose of robot        
            metadata = c.last_event.events[agent_id].metadata
            location = {
                "x": metadata["agent"]["position"]["x"],
                "y": metadata["agent"]["position"]["y"],
                "z": metadata["agent"]["position"]["z"],
                "rotation": metadata["agent"]["rotation"]["y"],
                "horizon": metadata["agent"]["cameraHorizon"]}
            
            prev_dist_goals[ia] = dist_goals[ia] # store the previous distance to goal
            dist_goals[ia] = distance_pts([location['x'], location['y'], location['z']], crp[ia])
            
            dist_del = abs(dist_goals[ia] - prev_dist_goals[ia])
            print (ia, "Dist to Goal: ", dist_goals[ia], dist_del, clost_node_location[ia])
            if dist_del < 0.2:
                # robot did not move 
                count_since_update[ia] += 1
            else:
                # robot moving 
                count_since_update[ia] = 0
                
            if count_since_update[ia] < 15:
                action_queue.append({'action':'ObjectNavExpertAction', 'position':dict(x=crp[ia][0], y=crp[ia][1], z=crp[ia][2]), 'agent_id':agent_id})
            else:    
                #updating goal
                clost_node_location[ia] += 1
                count_since_update[ia] = 0
                crp = closest_node(dest_obj_pos, reachable_positions, no_agents, clost_node_location)
    
            time.sleep(0.5)

    # align the robot once goal is reached
    # compute angle between robot heading and object
    metadata = c.last_event.events[agent_id].metadata
    robot_location = {
        "x": metadata["agent"]["position"]["x"],
        "y": metadata["agent"]["position"]["y"],
        "z": metadata["agent"]["position"]["z"],
        "rotation": metadata["agent"]["rotation"]["y"],
        "horizon": metadata["agent"]["cameraHorizon"]}
    
    robot_object_vec = [dest_obj_pos[0] -robot_location['x'], dest_obj_pos[2]-robot_location['z']]
    y_axis = [0, 1]
    unit_y = y_axis / np.linalg.norm(y_axis)
    unit_vector = robot_object_vec / np.linalg.norm(robot_object_vec)
    
    angle = math.atan2(np.linalg.det([unit_vector,unit_y]),np.dot(unit_vector,unit_y))
    angle = 360*angle/(2*np.pi)
    angle = (angle + 360) % 360
    rot_angle = angle - robot_location['rotation']
    
    if rot_angle > 0:
        action_queue.append({'action':'RotateRight', 'degrees':abs(rot_angle), 'agent_id':agent_id})
    else:
        action_queue.append({'action':'RotateLeft', 'degrees':abs(rot_angle), 'agent_id':agent_id})
        
    print ("Reached: ", dest_obj)
    
def PickupObject(robot, pick_obj):
    robot_name = robot['name']
    agent_id = int(robot_name[-1]) - 1
    objs = list(set([obj["objectId"] for obj in c.last_event.metadata["objects"]]))
    
    for obj in objs:
        match = re.match(pick_obj, obj)
        if match is not None:
            pick_obj_id = obj
            break # find the first instance
        
    action_queue.append({'action':'PickupObject', 'objectId':pick_obj_id, 'agent_id':agent_id})
    
def PutObject(robot, put_obj, recp):
    robot_name = robot['name']
    agent_id = int(robot_name[-1]) - 1
    objs = list(set([obj["objectId"] for obj in c.last_event.metadata["objects"]]))
    objs_center = list([obj["axisAlignedBoundingBox"]["center"] for obj in c.last_event.metadata["objects"]])
    objs_dists = list([obj["distance"] for obj in c.last_event.metadata["objects"]])

    metadata = c.last_event.events[agent_id].metadata
    robot_location = [metadata["agent"]["position"]["x"], metadata["agent"]["position"]["y"], metadata["agent"]["position"]["z"]]
    dist_to_recp = 9999999 # distance b/w robot and the recp obj
    for idx, obj in enumerate(objs):
        match = re.match(recp, obj)
        if match is not None:
            dist = objs_dists[idx]# distance_pts(robot_location, [objs_center[idx]['x'], objs_center[idx]['y'], objs_center[idx]['z']])
            if dist < dist_to_recp:
                recp_obj_id = obj
                dest_obj_center = objs_center[idx]
                dist_to_recp = dist
    action_queue.append({'action':'PutObject', 'objectId':recp_obj_id, 'agent_id':agent_id})
         
def SwitchOn(robot, sw_obj):
    robot_name = robot['name']
    agent_id = int(robot_name[-1]) - 1
    objs = list(set([obj["objectId"] for obj in c.last_event.metadata["objects"]]))
    
    for obj in objs:
        match = re.match(sw_obj, obj)
        if match is not None:
            sw_obj_id = obj
            break # find the first instance
    
    action_queue.append({'action':'ToggleObjectOn', 'objectId':sw_obj_id, 'agent_id':agent_id})      
        
def SwitchOff(robot, sw_obj):
    robot_name = robot['name']
    agent_id = int(robot_name[-1]) - 1
    objs = list(set([obj["objectId"] for obj in c.last_event.metadata["objects"]]))
    
    for obj in objs:
        match = re.match(sw_obj, obj)
        if match is not None:
            sw_obj_id = obj
            break # find the first instance
    
    action_queue.append({'action':'ToggleObjectOff', 'objectId':sw_obj_id, 'agent_id':agent_id})        

def OpenObject(robot, sw_obj):
    robot_name = robot['name']
    agent_id = int(robot_name[-1]) - 1
    objs = list(set([obj["objectId"] for obj in c.last_event.metadata["objects"]]))
    
    for obj in objs:
        match = re.match(sw_obj, obj)
        if match is not None:
            sw_obj_id = obj
            break # find the first instance
    
    action_queue.append({'action':'OpenObject', 'objectId':sw_obj_id, 'agent_id':agent_id})
    
def CloseObject(robot, sw_obj):
    robot_name = robot['name']
    agent_id = int(robot_name[-1]) - 1
    objs = list(set([obj["objectId"] for obj in c.last_event.metadata["objects"]]))
    
    for obj in objs:
        match = re.match(sw_obj, obj)
        if match is not None:
            sw_obj_id = obj
            break # find the first instance
    
    action_queue.append({'action':'CloseObject', 'objectId':sw_obj_id, 'agent_id':agent_id}) 
    
def BreakObject(robot, sw_obj):
    robot_name = robot['name']
    agent_id = int(robot_name[-1]) - 1
    objs = list(set([obj["objectId"] for obj in c.last_event.metadata["objects"]]))
    
    for obj in objs:
        match = re.match(sw_obj, obj)
        if match is not None:
            sw_obj_id = obj
            break # find the first instance
    
    action_queue.append({'action':'BreakObject', 'objectId':sw_obj_id, 'agent_id':agent_id}) 
    
def SliceObject(robot, sw_obj):
    robot_name = robot['name']
    agent_id = int(robot_name[-1]) - 1
    objs = list(set([obj["objectId"] for obj in c.last_event.metadata["objects"]]))
    
    for obj in objs:
        match = re.match(sw_obj, obj)
        if match is not None:
            sw_obj_id = obj
            break # find the first instance
    
    action_queue.append({'action':'SliceObject', 'objectId':sw_obj_id, 'agent_id':agent_id})      
  
def CleanObject(robot, sw_obj):
    robot_name = robot['name']
    agent_id = int(robot_name[-1]) - 1
    objs = list(set([obj["objectId"] for obj in c.last_event.metadata["objects"]]))

    for obj in objs:
        match = re.match(sw_obj, obj)
        if match is not None:
            sw_obj_id = obj
            break # find the first instance

    action_queue.append({'action':'CleanObject', 'objectId':sw_obj_id, 'agent_id':agent_id}) 

### **🤖 机器人任务执行**
def execute_action(act, img_counter):
    act = action_queue[0]
    try:
        if act['action'] == 'ObjectNavExpertAction':
            multi_agent_event = c.step(dict(action=act['action'], position=act['position'], agentId=act['agent_id']))
            next_action = multi_agent_event.metadata['actionReturn']

            if next_action != None:
                multi_agent_event = c.step(action=next_action, agentId=act['agent_id'], forceAction=True)
        
        elif act['action'] == 'MoveAhead':
            multi_agent_event = c.step(action="MoveAhead", agentId=act['agent_id'])
            
        elif act['action'] == 'MoveBack':
            multi_agent_event = c.step(action="MoveBack", agentId=act['agent_id'])
                
        elif act['action'] == 'RotateLeft':
            multi_agent_event = c.step(action="RotateLeft", degrees=act['degrees'], agentId=act['agent_id'])
            
        elif act['action'] == 'RotateRight':
            multi_agent_event = c.step(action="RotateRight", degrees=act['degrees'], agentId=act['agent_id'])
            
        elif act['action'] == 'PickupObject':
            multi_agent_event = c.step(action="PickupObject", objectId=act['objectId'], agentId=act['agent_id'], forceAction=True) 

        elif act['action'] == 'PutObject':
            multi_agent_event = c.step(action="PutObject", objectId=act['objectId'], agentId=act['agent_id'], forceAction=True)

        elif act['action'] == 'ToggleObjectOn':
            multi_agent_event = c.step(action="ToggleObjectOn", objectId=act['objectId'], agentId=act['agent_id'], forceAction=True)
        
        elif act['action'] == 'ToggleObjectOff':
            multi_agent_event = c.step(action="ToggleObjectOff", objectId=act['objectId'], agentId=act['agent_id'], forceAction=True)
        
        elif act['action'] == 'Done':
            multi_agent_event = c.step(action="Done")
    
    except Exception as e:
        print (e)

    for i,e in enumerate(multi_agent_event.events):
        cv2.imshow('agent%s' % i, e.cv2img)
        f_name = os.path.dirname(__file__) + "/agent_" + str(i+1) + "/img_" + str(img_counter).zfill(5) + ".png"
        cv2.imwrite(f_name, e.cv2img)
    top_view_rgb = cv2.cvtColor(c.last_event.events[0].third_party_camera_frames[-1], cv2.COLOR_BGR2RGB)
    cv2.imshow('Top View', top_view_rgb)
    f_name = os.path.dirname(__file__) + "/top_view/img_" + str(img_counter).zfill(5) + ".png"
    cv2.imwrite(f_name, e.cv2img)

def task_execution_loop():
    """
    机器人执行任务队列
    """
    # delete if current output already exist
    cur_path = os.path.dirname(__file__) + "/*/"
    for x in glob(cur_path, recursive = True):
        shutil.rmtree (x)
    
    # create new folders to save the images from the agents
    for i in range(no_robot):
        folder_name = "agent_" + str(i+1)
        folder_path = os.path.dirname(__file__) + "/" + folder_name
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    
    # create folder to store the top view images
    folder_name = "top_view"
    folder_path = os.path.dirname(__file__) + "/" + folder_name
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    img_counter = 0
    while not task_over:
        if len(action_queue) > 0:
            act = action_queue.pop(0)
            execute_action(act,img_counter )
            time.sleep(0.5)  # 避免过载
        else:
            break
        img_counter += 1    
    print("All tasks completed!", img_counter)

### **🚀 运行主任务**
def main():
    # 1. **获取物体信息**
    env_objects = get_environment_data(controller=c)

    # 2. **构建环境图**
    env_graph = build_environment_graph(env_objects)
    print(env_graph)

    # 3. **生成任务序列（使用 Graphormer 安全感知）**
    task_description = "切苹果"
    task_sequence_json = generate_task_sequence(task_description, env_objects)
    print(f"Generated Task Sequence:\n{task_sequence_json}")
    global action_queue
    action_queue =  parse_task_sequence(task_sequence_json)
    print(action_queue)

    print("starting executing!")
    # 4. **执行任务**
    task_execution_thread = threading.Thread(target=task_execution_loop)
    task_execution_thread.start()

if __name__ == "__main__":
    main()
