import math
from ai2thor.controller import Controller
import numpy as np
import threading
import shutil
import cv2
import random
import os
from scipy.spatial import distance
from glob import glob
import time
import re

# VERY IMPORTANT !!!
# This is the file save dir. if you delete this var, the script will remove ALL subdir in your folder
# For your data safety, I have disable the lines to use shutil to rmdir :)
__file__ = './testing'

# L^2 distance
def distance_pts(p1, p2):
    return ((p1['x'] - p2['x']) ** 2 + (p1['y'] - p2['y']) ** 2 +(p1['z'] - p2['z']) ** 2) ** 0.5

def distance(a, b):
    return math.sqrt((a['x'] - b["x"]) ** 2 + (a['z'] - b["z"]) ** 2)

# path planing: very simple a_star
def a_star_planning(start, goal, reachable_positions):
    path = [start]
    
    while distance(path[-1], goal) > 0.5: 
        next_pos = min(reachable_positions, key=lambda p: distance(p, goal) + distance(p, path[-1]))
        if next_pos in path:
            break  
        path.append(next_pos)

    path.append(goal)  
    return path


class ControlPolicy:
    def __init__(self,controller,tag):
        self.c = controller
        self.action_queue = []
        self.task_over = False
        self.robots = []
        self.robot = None # for single robot system
        self.handle_safty_issue_targets = []
        self.tag = tag+"_"
        self.image_save_path = __file__


    def init_robots(self, robots):
        self.robots = robots
        self.robot = robots[0]
        no_robot = len(robots)
        self.no_robot = no_robot

        # initialize n agents into the scene
        multi_agent_event = self.c.step(dict(action='Initialize', agentMode="default", snapGrid=False, gridSize=0.25, rotateStepDegrees=20, visibilityDistance=100, fieldOfView=90, agentCount=no_robot))

        # add a top view camera
        event = self.c.step(action="GetMapViewCameraProperties")
        event = self.c.step(action="AddThirdPartyCamera", **event.metadata["actionReturn"])

        # get reachabel positions
        reachable_positions_ = self.c.step(action="GetReachablePositions").metadata["actionReturn"]
        self.reachable_positions = reachable_positions_
        # self.reachable_positions = positions_tuple = [(p["x"], p["y"], p["z"]) for p in reachable_positions_]

        # initialize robots
        for i in range(no_robot):
            init_pos = random.choice(reachable_positions_)
            self.c.step(dict(action="Teleport", position=init_pos, agentId=i))

    # Automaticall add LLM output actions to action list in controller
    def add_action_list(self,action_list):
        # check type
        if not isinstance(action_list, list):
            action_list = [action_list]
        # add action
        for act in action_list:
            if act['action'] == "GoToObject":
                self.GoToObject(self.robot,act['object_id'],self.reachable_positions)
            if act['action'] == 'PickupObject':
                self.PickupObject(self.robot,act['object_id'])
            if act['action'] == 'PutObject':
                self.PutObject(self.robot,act['object_id'], act['target_id'] if 'target_id' in act else act['object_id'])
            if act['action'] == 'SwitchOn':
                self.SwitchOn(self.robot,act['object_id'])
            if act['action'] == 'SwitchOff':
                self.SwitchOff(self.robot,act['object_id'])
            if act['action'] == 'HandleSafetyIssue':
                self.HandleSafetyIssue(self.robot,act['object_id'])
            if act['action'] == "Done":
                self.action_queue.append({'action':'Done'})
        # print(self.action_queue)

    # manually set action_queue for testing
    def set_action_queue(self, action_queue):
        self.action_queue = action_queue
    
    # action HandleSafetyIssue
    def HandleSafetyIssue(self,robot,target_name):
        self.action_queue.append({'action':'HandleSafetyIssue', 'target':target_name, 'agent_id':robot})

    # GoToObject: Navigate agent to a target object
    def GoToObject(self, robot, dest_obj, reachable_positions):
        robot_name = robot['name']
        agent_id = int(robot_name[-1]) - 1
        print(f"Going to {dest_obj} (Agent {agent_id})")

        # Ignore if the destination is "Baby" (simulation logic)
        if "Baby" in dest_obj:
            return

        # Retrieve object metadata
        objects_metadata = self.c.last_event.metadata["objects"]
        objs = {obj["name"]: obj for obj in objects_metadata}

        if dest_obj not in objs:
            raise Exception(f"Object '{dest_obj}' not found!")
        
        metadata = self.c.last_event.events[agent_id].metadata
        robot_location = metadata["agent"]["position"]
        robot_rotation = metadata["agent"]["rotation"]["y"]

        dest_obj_data = objs[dest_obj]
        dest_obj_pos = dest_obj_data["position"]
        closest_goal = min(reachable_positions, key=lambda p: distance(p, dest_obj_pos))

        # Teleport along path
        path = a_star_planning(robot_location, closest_goal, reachable_positions)

        for i, waypoint in enumerate(path):
            print(f"Teleporting {i+1}/{len(path)}: {waypoint}")

            self.action_queue.append({
                'action' : "Teleport",
                'position': waypoint,
                'agent_id':agent_id
            })
    

        # Rotate robot to face the object
        robot_object_vec = np.array([dest_obj_pos["x"] -closest_goal['x'], dest_obj_pos["z"] - closest_goal['z']])
        y_axis = np.array([0, 1])


        unit_vector = robot_object_vec / np.linalg.norm(robot_object_vec)
        unit_y = y_axis / np.linalg.norm(y_axis)

        angle = math.degrees(math.atan2(np.linalg.det([unit_vector, unit_y]), np.dot(unit_vector, unit_y)))
        angle = (angle + 360) % 360
        rot_angle = angle - robot_rotation

        self.action_queue.append({
            "action": "RotateRight", # if rot_angle > 0 else "RotateLeft",
            "degrees": abs(rot_angle),
            "agent_id": agent_id
        })


        
    def PickupObject(self,robot, pick_obj):
        robot_name = robot['name']
        agent_id = int(robot_name[-1]) - 1
        # objs = set([obj["name"] for obj in self.c.last_event.metadata["objects"]])
        objs_ids = {obj['name']:obj['objectId'] for obj in self.c.last_event.metadata["objects"]}
        pick_obj_id = objs_ids[pick_obj]
            
        self.action_queue.append({'action':'PickupObject', 'objectId':pick_obj_id, 'agent_id':agent_id})
        
    def PutObject(self,robot, put_obj, recp):
        robot_name = robot['name']
        agent_id = int(robot_name[-1]) - 1
        objs = list([obj["name"] for obj in self.c.last_event.metadata["objects"]])
        objs_ids = {obj['name']:obj['objectId'] for obj in self.c.last_event.metadata["objects"]}
        objs_center = list([obj["axisAlignedBoundingBox"]["center"] for obj in self.c.last_event.metadata["objects"]])
        objs_dists = list([obj["distance"] for obj in self.c.last_event.metadata["objects"]])

        recp_obj_id = objs_ids[recp]

        self.action_queue.append({'action':'PutObject', 'objectId':recp_obj_id, 'agent_id':agent_id})
            
    def SwitchOn(self,robot, sw_obj):
        robot_name = robot['name']
        agent_id = int(robot_name[-1]) - 1
        objs_ids = {obj['name']:obj['objectId'] for obj in self.c.last_event.metadata["objects"]}
        sw_obj_id= objs_ids[sw_obj]
        
        self.action_queue.append({'action':'ToggleObjectOn', 'objectId':sw_obj_id, 'agent_id':agent_id})      
            
    def SwitchOff(self,robot, sw_obj):
        robot_name = robot['name']
        agent_id = int(robot_name[-1]) - 1
        objs_ids = {obj['name']:obj['objectId'] for obj in self.c.last_event.metadata["objects"]}
        sw_obj_id= objs_ids[sw_obj]
        
        self.action_queue.append({'action':'ToggleObjectOff', 'objectId':sw_obj_id, 'agent_id':agent_id})        

    def OpenObject(self,robot, sw_obj):
        robot_name = robot['name']
        agent_id = int(robot_name[-1]) - 1
        objs_ids = {obj['name']:obj['objectId'] for obj in self.c.last_event.metadata["objects"]}
        sw_obj_id= objs_ids[sw_obj]
        self.action_queue.append({'action':'OpenObject', 'objectId':sw_obj_id, 'agent_id':agent_id})
        
    def CloseObject(self,robot, sw_obj):
        robot_name = robot['name']
        agent_id = int(robot_name[-1]) - 1
        objs_ids = {obj['name']:obj['objectId'] for obj in self.c.last_event.metadata["objects"]}
        sw_obj_id= objs_ids[sw_obj]
        
        self.action_queue.append({'action':'CloseObject', 'objectId':sw_obj_id, 'agent_id':agent_id}) 
        
    def BreakObject(self,robot, sw_obj):
        robot_name = robot['name']
        agent_id = int(robot_name[-1]) - 1
        objs_ids = {obj['name']:obj['objectId'] for obj in self.c.last_event.metadata["objects"]}
        sw_obj_id= objs_ids[sw_obj]
        
        self.action_queue.append({'action':'BreakObject', 'objectId':sw_obj_id, 'agent_id':agent_id}) 
        
    def SliceObject(self,robot, sw_obj):
        robot_name = robot['name']
        agent_id = int(robot_name[-1]) - 1
        objs_ids = {obj['name']:obj['objectId'] for obj in self.c.last_event.metadata["objects"]}
        sw_obj_id= objs_ids[sw_obj]
        
        self.action_queue.append({'action':'SliceObject', 'objectId':sw_obj_id, 'agent_id':agent_id})      
    
    def CleanObject(self,robot, sw_obj):
        robot_name = robot['name']
        agent_id = int(robot_name[-1]) - 1
        objs_ids = {obj['name']:obj['objectId'] for obj in self.c.last_event.metadata["objects"]}
        sw_obj_id= objs_ids[sw_obj]

        self.action_queue.append({'action':'CleanObject', 'objectId':sw_obj_id, 'agent_id':agent_id}) 

    # execute an action and save images
    def execute_action(self, act, img_counter):
        try:
            if act['action'] == 'ObjectNavExpertAction':
                multi_agent_event = self.c.step(dict(action=act['action'], position=act['position'], agentId=act['agent_id']))
                next_action = multi_agent_event.metadata['actionReturn']
                if next_action != None:
                    multi_agent_event = self.c.step(action=next_action, agentId=act['agent_id'], forceAction=True)
            elif act['action'] =='GetShortestPath':
                multi_agent_event = self.c.step(action="GetShortestPath", agentId= act['agent_id'], position=act['position'])
            elif act['action'] =='Teleport':
                multi_agent_event = self.c.step(action="Teleport", agentId= act['agent_id'], position=act['position'])
            elif act['action'] == 'MoveAhead':
                multi_agent_event = self.c.step(action="MoveAhead", agentId=act['agent_id'])
                
            elif act['action'] == 'MoveBack':
                multi_agent_event = self.c.step(action="MoveBack", agentId=act['agent_id'])
                    
            elif act['action'] == 'RotateLeft':
                multi_agent_event = self.c.step(action="RotateLeft", degrees=act['degrees'], agentId=act['agent_id'])
                
            elif act['action'] == 'RotateRight':
                multi_agent_event = self.c.step(action="RotateRight", degrees=act['degrees'], agentId=act['agent_id'])
                
            elif act['action'] == 'PickupObject':
                multi_agent_event = self.c.step(action="PickupObject", objectId=act['objectId'], agentId=act['agent_id'], forceAction=True) 

            elif act['action'] == 'PutObject':
                multi_agent_event = self.c.step(action="PutObject", objectId=act['objectId'], agentId=act['agent_id'], forceAction=True)

            elif act['action'] == 'ToggleObjectOn':
                multi_agent_event = self.c.step(action="ToggleObjectOn", objectId=act['objectId'], agentId=act['agent_id'], forceAction=True)
            
            elif act['action'] == 'ToggleObjectOff':
                multi_agent_event = self.c.step(action="ToggleObjectOff", objectId=act['objectId'], agentId=act['agent_id'], forceAction=True)
            
            elif act['action'] == 'HandleSafetyIssue':
                self.handle_safty_issue_targets.append(act['target'])
                return
            
            elif act['action'] == 'Done':
                self.task_over = True
                # multi_agent_event = self.c.step(action="Done")
                return
        
        except Exception as e:
            print (e)
            return

        for i,e in enumerate(multi_agent_event.events):
            cv2.imshow('agent%s' % i, e.cv2img)
            f_name = self.image_save_path+'//' + self.tag+ "agent_" + str(i+1) + "/img_" + str(img_counter).zfill(5) + ".png"
            cv2.imwrite(f_name, e.cv2img)
        # print(len(self.c.last_event.events[0].third_party_camera_frames))
        top_view = self.c.last_event.events[0].third_party_camera_frames[-1]
        top_view_bgr = cv2.cvtColor(top_view, cv2.COLOR_RGB2BGR)
        cv2.imshow('Top View', top_view)
        f_name = self.image_save_path+'//'+self.tag+ "top_view/img_" + str(img_counter).zfill(5) + ".png"
        cv2.imwrite(f_name, top_view_bgr)

    def task_execution_loop(self):
        """
        execute tasks
        """
        # delete if current output already exist
        # cur_path = self.image_save_path + f"/{tag}/"
        # for x in glob(cur_path, recursive = True):
        #     shutil.rmtree (x)
        
        # create new folders to save the images from the agents
        for i in range(self.no_robot):
            folder_name = self.tag+"agent_" + str(i+1)
            folder_path = self.image_save_path + "/" + folder_name
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
        
        # create folder to store the top view images
        folder_name = self.tag+"top_view"
        folder_path =  self.image_save_path + "/" + folder_name
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        img_counter = 0
        self.task_over = False
        while not self.task_over:
            if len(self.action_queue) > 0:
                print('exec')
                act = self.action_queue.pop(0)
                self.execute_action(act,img_counter)
                time.sleep(0.5) 
            else:
                self.task_over = True
                print("All tasks completed!", img_counter)
                break
            img_counter += 1    
    def run_task_thread(self):
        print(self.action_queue)
        task_execution_thread = threading.Thread(target=self.task_execution_loop)
        task_execution_thread.start()


if __name__ == "__main__":
    __file__ = './testing'
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
    test_actions = [{'action': 'GoToObject', 'object_id': 'Apple_3fef4551'}, {'action': 'PickupObject', 'object_id': 'Apple_3fef4551'}, {'action': 'GoToObject', 'object_id': 'Bowl_208f368b'}, {'action': 'PutObject', 'object_id': 'Apple_3fef4551', 'target_id': 'Bowl_208f368b'}, {'action': 'Done'}]
    cp.add_action_list(test_actions)
    cp.run_task_thread()
