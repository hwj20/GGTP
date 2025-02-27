import shutil
import cv2
import random
import os
import numpy as np
from scipy.spatial import distance
from glob import glob
import time
import re
import math

# VERY IMPORTANT !!!
# VERY IMPORTANT !!!
# VERY IMPORTANT !!!
# This is the file save path. if you delete this var, the script will remove ALL subdir in your folder
__file__ = './testing'

def distance_pts(p1, p2):
    # print(p1,p2)
    return ((p1['x'] - p2['x']) ** 2 + (p1['y'] - p2['y']) ** 2 +(p1['z'] - p2['z']) ** 2) ** 0.5

def closest_node(node, nodes, no_robot, clost_node_location):
    crps = []
    distances = distance.cdist([node], nodes)[0]
    dist_indices = np.argsort(np.array(distances))
    for i in range(no_robot):
        pos_index = dist_indices[(i * 5) + clost_node_location[i]]
        crps.append (nodes[pos_index])
    return crps

class ControlPolicy:
    def __init__(self,controller):
        self.c = controller
        self.action_queue = []
        self.task_over = False
        self.robots = []
        self.robot = None # for single robot system

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
        self.reachable_positions = positions_tuple = [(p["x"], p["y"], p["z"]) for p in reachable_positions_]

        # initialize robots
        for i in range(no_robot):
            init_pos = random.choice(reachable_positions_)
            self.c.step(dict(action="Teleport", position=init_pos, agentId=i))

    def add_action_list(self,action_list):
        if not isinstance(action_list, list):
            action_list = [action_list]

        for act in action_list:
            if act['action'] == "GoToObject":
                self.GoToObject(self.robots,act['object_id'],self.reachable_positions)
            if act['action'] == 'PickupObjet':
                self.PickupObject(self.robot,act['object_id'])
            if act['action'] == 'PutObjet':
                self.PutObject(self.robot,act['object_id'])
            if act['action'] == 'SwitchOn':
                self.SwitchOn(self.robot,act['object_id'])
            if act['action'] == 'SwitchOff':
                self.SwitchOff(self.robot,act['object_id'])
            if act['action'] == "Done":
                self.action_queue.append({'action':'Done'})
        print(self.action_queue)
        
    def GoToObject(self, robots, dest_obj, reachable_positions):
        print("Going to", dest_obj)
        
        if not isinstance(robots, list):
            robots = [robots]
        
        no_agents = len(robots)
        count_since_update = [0] * no_agents
        closest_node_location = [0] * no_agents
        
        # 获取场景中的对象信息
        objects_metadata = self.c.last_event.metadata["objects"]
        objs = {obj["name"]: obj for obj in objects_metadata}
        
        if dest_obj not in objs:
            raise Exception("Object not found")
        
        dest_obj_data = objs[dest_obj]
        dest_obj_center = dest_obj_data["axisAlignedBoundingBox"]["center"]
        dest_obj_pos = [dest_obj_center['x'], dest_obj_center['y'], dest_obj_center['z']]
        
        # 计算机器人最接近目标物体的可达位置
        crp = closest_node(dest_obj_pos, reachable_positions, no_agents, closest_node_location)
        
        goal_thresh = 0.3
        robots_reached_goal = [False] * no_agents
        locations = []
        for ia in range(self.no_robot):
            metadata = self.c.last_event.events[ia].metadata
            location = metadata["agent"]["position"]
            locations.append(location)
        
        while not all(robots_reached_goal):
            for ia, robot in enumerate(robots):
                agent_id = int(robot['name'][-1]) - 1
                
                location = locations[ia]
                
                current_distance = distance_pts(location, {"x": crp[ia][0], "y": crp[ia][1], "z": crp[ia][2]})
                
                if current_distance <= goal_thresh:
                    robots_reached_goal[ia] = True
                    continue
                
                if count_since_update[ia] < 15:
                    self.action_queue.append({
                        # 'action': 'ObjectNavExpertAction',
                        'action': 'Teleport',
                        'position': dict(x=crp[ia][0], y=crp[ia][1], z=crp[ia][2]),
                        'agent_id': agent_id
                    })
                    locations[ia] = {"x": crp[ia][0], "y": crp[ia][1], "z": crp[ia][2]}
                else:
                    # 更新目标点
                    closest_node_location[ia] += 1
                    count_since_update[ia] = 0
                    crp = closest_node(dest_obj_pos, reachable_positions, no_agents, closest_node_location)
        
        # 机器人朝向目标物体
        metadata = self.c.last_event.events[agent_id].metadata
        robot_location = metadata["agent"]["position"]
        robot_rotation = metadata["agent"]["rotation"]["y"]
        
        robot_object_vec = np.array([dest_obj_pos[0] - robot_location['x'], dest_obj_pos[2] - robot_location['z']])
        y_axis = np.array([0, 1])
        
        unit_vector = robot_object_vec / np.linalg.norm(robot_object_vec)
        unit_y = y_axis / np.linalg.norm(y_axis)
        
        angle = math.degrees(math.atan2(np.linalg.det([unit_vector, unit_y]), np.dot(unit_vector, unit_y)))
        angle = (angle + 360) % 360
        rot_angle = angle - robot_rotation
        
        self.action_queue.append({
            'action': 'RotateRight' if rot_angle > 0 else 'RotateLeft',
            'degrees': abs(rot_angle),
            'agent_id': agent_id
        })
        
        print("Reached:", dest_obj)


    # def GoToObject(self,robots, dest_obj, reachable_positions):
    #     print ("Going to ", dest_obj)
    #     # check if robots is a list
        
    #     if not isinstance(robots, list):
    #         # convert robot to a list
    #         robots = [robots]
    #     no_agents = len (robots)
    #     # robots distance to the goal 
    #     dist_goals = [10.0] * len(robots)
    #     prev_dist_goals = [10.0] * len(robots)
    #     count_since_update = [0] * len(robots)
    #     clost_node_location = [0] * len(robots)
        
    #     # list of objects in the scene and their centers
    #     objs = list([obj["name"] for obj in self.c.last_event.metadata["objects"]])
    #     objs_ids = {obj['name']:obj['objectId'] for obj in self.c.last_event.metadata["objects"]}
    #     objs_center = list([obj["axisAlignedBoundingBox"]["center"] for obj in self.c.last_event.metadata["objects"]])
                
    #     dest_obj_center = None

    #     # look for the location and id of the destination object
    #     for idx, obj in enumerate(objs):
    #         match = re.match(dest_obj, obj)
    #         if match is not None:
    #             dest_obj_id = objs_ids[obj]
    #             dest_obj_center = objs_center[idx]
    #             break # find the first instance

    #     if dest_obj_center == None:
    #         raise Exception("not found")
    #     dest_obj_pos = [dest_obj_center['x'], dest_obj_center['y'], dest_obj_center['z']] 
        
        
    #     # closest reachable position for each robot
    #     # all robots cannot reach the same spot 
    #     # differt close points needs to be found for each robot
    #     crp = closest_node(dest_obj_pos, reachable_positions, no_agents, clost_node_location)
        
    #     goal_thresh = 0.3
    #     # at least one robot is far away from the goal
        
    #     while all(d > goal_thresh for d in dist_goals):
    #         for ia, robot in enumerate(robots):
    #             robot_name = robot['name']
    #             agent_id = int(robot_name[-1]) - 1
                
    #             # get the pose of robot        
    #             metadata = self.c.last_event.events[agent_id].metadata
    #             location = {
    #                 "x": metadata["agent"]["position"]["x"],
    #                 "y": metadata["agent"]["position"]["y"],
    #                 "z": metadata["agent"]["position"]["z"],
    #                 "rotation": metadata["agent"]["rotation"]["y"],
    #                 "horizon": metadata["agent"]["cameraHorizon"]}
                
    #             prev_dist_goals[ia] = dist_goals[ia] # store the previous distance to goal
    #             dist_goals[ia] = distance_pts({'x':location['x'], 'y':location['y'],'z': location['z']}, 
    #                                           {'x':crp[ia][0],'y':crp[ia][1],'z':crp[ia][2]})
                
    #             dist_del = abs(dist_goals[ia] - prev_dist_goals[ia])
    #             # print (ia, "Dist to Goal: ", dist_goals[ia], dist_del, clost_node_location[ia])
    #             if dist_del < 0.2:
    #                 # robot did not move 
    #                 count_since_update[ia] += 1
    #             else:
    #                 # robot moving 
    #                 count_since_update[ia] = 0
                    
    #             if count_since_update[ia] < 15:
    #                 self.action_queue.append({'action':'ObjectNavExpertAction', 'position':dict(x=crp[ia][0], y=crp[ia][1], z=crp[ia][2]), 'agent_id':agent_id})
    #             else:    
    #                 #updating goal
    #                 clost_node_location[ia] += 1
    #                 count_since_update[ia] = 0
    #                 crp = closest_node(dest_obj_pos, reachable_positions, no_agents, clost_node_location)
        
    #             # time.sleep(0.5)

    #     # align the robot once goal is reached
    #     # compute angle between robot heading and object
    #     metadata = self.c.last_event.events[agent_id].metadata
    #     robot_location = {
    #         "x": metadata["agent"]["position"]["x"],
    #         "y": metadata["agent"]["position"]["y"],
    #         "z": metadata["agent"]["position"]["z"],
    #         "rotation": metadata["agent"]["rotation"]["y"],
    #         "horizon": metadata["agent"]["cameraHorizon"]}
        
    #     robot_object_vec = [dest_obj_pos[0] -robot_location['x'], dest_obj_pos[2]-robot_location['z']]
    #     y_axis = [0, 1]
    #     unit_y = y_axis / np.linalg.norm(y_axis)
    #     unit_vector = robot_object_vec / np.linalg.norm(robot_object_vec)
        
    #     angle = math.atan2(np.linalg.det([unit_vector,unit_y]),np.dot(unit_vector,unit_y))
    #     angle = 360*angle/(2*np.pi)
    #     angle = (angle + 360) % 360
    #     rot_angle = angle - robot_location['rotation']
        
    #     if rot_angle > 0:
    #         self.action_queue.append({'action':'RotateRight', 'degrees':abs(rot_angle), 'agent_id':agent_id})
    #     else:
    #         self.action_queue.append({'action':'RotateLeft', 'degrees':abs(rot_angle), 'agent_id':agent_id})
            
    #     print ("Reached: ", dest_obj)
        
    def PickupObject(self,robot, pick_obj):
        robot_name = robot['name']
        agent_id = int(robot_name[-1]) - 1
        objs = set([obj["name"] for obj in self.c.last_event.metadata["objects"]])
        objs_ids = {obj['name']:obj['objecId'] for obj in self.c.last_event.metadata["objects"]}
        pick_obj_id = objs_ids[objs[pick_obj]]
        # for obj in objs:
        #     match = re.match(pick_obj, obj)
        #     if match is not None:
        #         pick_obj_id = obj
        #         break # find the first instance
            
        self.action_queue.append({'action':'PickupObject', 'objectId':pick_obj_id, 'agent_id':agent_id})
        
    def PutObject(self,robot, put_obj, recp):
        robot_name = robot['name']
        agent_id = int(robot_name[-1]) - 1
        objs = list([obj["name"] for obj in self.c.last_event.metadata["objects"]])
        objs_ids = {obj['name']:obj['object_id'] for obj in self.c.last_event.metadata["objects"]}
        objs_center = list([obj["axisAlignedBoundingBox"]["center"] for obj in c.last_event.metadata["objects"]])
        objs_dists = list([obj["distance"] for obj in self.c.last_event.metadata["objects"]])

        metadata = self.c.last_event.events[agent_id].metadata
        robot_location = [metadata["agent"]["position"]["x"], metadata["agent"]["position"]["y"], metadata["agent"]["position"]["z"]]
        dist_to_recp = 9999999 # distance b/w robot and the recp obj
        for idx, obj in enumerate(objs):
            match = re.match(recp, obj)
            if match is not None:
                dist = objs_dists[idx]# distance_pts(robot_location, [objs_center[idx]['x'], objs_center[idx]['y'], objs_center[idx]['z']])
                if dist < dist_to_recp:
                    recp_obj_id =objs_ids[obj]
                    dest_obj_center = objs_center[idx]
                    dist_to_recp = dist
        self.action_queue.append({'action':'PutObject', 'objectId':recp_obj_id, 'agent_id':agent_id})
            
    def SwitchOn(self,robot, sw_obj):
        robot_name = robot['name']
        agent_id = int(robot_name[-1]) - 1
        # objs = set([obj["name"] for obj in self.c.last_event.metadata["objects"]])
        objs_ids = {obj['name']:obj['objectId'] for obj in self.c.last_event.metadata["objects"]}
        sw_obj_id= objs_ids[sw_obj]
        
        # for obj in objs:
        #     match = re.match(sw_obj, obj)
        #     if match is not None:
        #         sw_obj_id = obj
        #         break # find the first instance
        
        self.action_queue.append({'action':'ToggleObjectOn', 'objectId':sw_obj_id, 'agent_id':agent_id})      
            
    def SwitchOff(self,robot, sw_obj):
        robot_name = robot['name']
        agent_id = int(robot_name[-1]) - 1
        # objs = set([obj["name"] for obj in self.c.last_event.metadata["objects"]])
        objs_ids = {obj['name']:obj['objectId'] for obj in self.c.last_event.metadata["objects"]}
        sw_obj_id= objs_ids[sw_obj]
        
        # for obj in objs:
        #     match = re.match(sw_obj, obj)
        #     if match is not None:
        #         sw_obj_id = obj
        #         break # find the first instance
        
        self.action_queue.append({'action':'ToggleObjectOff', 'objectId':sw_obj_id, 'agent_id':agent_id})        

    def OpenObject(self,robot, sw_obj):
        robot_name = robot['name']
        agent_id = int(robot_name[-1]) - 1
        objs = list(set([obj["objectId"] for obj in self.c.last_event.metadata["objects"]]))
        
        for obj in objs:
            match = re.match(sw_obj, obj)
            if match is not None:
                sw_obj_id = obj
                break # find the first instance
        
        self.action_queue.append({'action':'OpenObject', 'objectId':sw_obj_id, 'agent_id':agent_id})
        
    def CloseObject(self,robot, sw_obj):
        robot_name = robot['name']
        agent_id = int(robot_name[-1]) - 1
        objs = list(set([obj["objectId"] for obj in self.c.last_event.metadata["objects"]]))
        
        for obj in objs:
            match = re.match(sw_obj, obj)
            if match is not None:
                sw_obj_id = obj
                break # find the first instance
        
        self.action_queue.append({'action':'CloseObject', 'objectId':sw_obj_id, 'agent_id':agent_id}) 
        
    def BreakObject(self,robot, sw_obj):
        robot_name = robot['name']
        agent_id = int(robot_name[-1]) - 1
        objs = list(set([obj["objectId"] for obj in self.c.last_event.metadata["objects"]]))
        
        for obj in objs:
            match = re.match(sw_obj, obj)
            if match is not None:
                sw_obj_id = obj
                break # find the first instance
        
        self.action_queue.append({'action':'BreakObject', 'objectId':sw_obj_id, 'agent_id':agent_id}) 
        
    def SliceObject(self,robot, sw_obj):
        robot_name = robot['name']
        agent_id = int(robot_name[-1]) - 1
        objs = list(set([obj["objectId"] for obj in self.c.last_event.metadata["objects"]]))
        
        for obj in objs:
            match = re.match(sw_obj, obj)
            if match is not None:
                sw_obj_id = obj
                break # find the first instance
        
        self.action_queue.append({'action':'SliceObject', 'objectId':sw_obj_id, 'agent_id':agent_id})      
    
    def CleanObject(self,robot, sw_obj):
        robot_name = robot['name']
        agent_id = int(robot_name[-1]) - 1
        objs = list(set([obj["objectId"] for obj in self.c.last_event.metadata["objects"]]))

        for obj in objs:
            match = re.match(sw_obj, obj)
            if match is not None:
                sw_obj_id = obj
                break # find the first instance

        self.action_queue.append({'action':'CleanObject', 'objectId':sw_obj_id, 'agent_id':agent_id}) 

    def execute_action(self, act, img_counter):
        try:
            if act['action'] == 'ObjectNavExpertAction':
                multi_agent_event = self.c.step(dict(action=act['action'], position=act['position'], agentId=act['agent_id']))
                next_action = multi_agent_event.metadata['actionReturn']

                if next_action != None:
                    multi_agent_event = self.c.step(action=next_action, agentId=act['agent_id'], forceAction=True)
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
            
            elif act['action'] == 'Done':
                self.task_over = True
                # multi_agent_event = self.c.step(action="Done")
                return
        
        except Exception as e:
            print (e)

        print('save')
        for i,e in enumerate(multi_agent_event.events):
            cv2.imshow('agent%s' % i, e.cv2img)
            f_name = __file__ + "/agent_" + str(i+1) + "/img_" + str(img_counter).zfill(5) + ".png"
            cv2.imwrite(f_name, e.cv2img)
        top_view_rgb = cv2.cvtColor(self.c.last_event.events[0].third_party_camera_frames[-1], cv2.COLOR_BGR2RGB)
        cv2.imshow('Top View', top_view_rgb)
        f_name = __file__ + "/top_view/img_" + str(img_counter).zfill(5) + ".png"
        cv2.imwrite(f_name, e.cv2img)

    def task_execution_loop(self):
        """
        机器人执行任务队列
        """
        # delete if current output already exist
        # cur_path = __file__ + "/*/"
        # for x in glob(cur_path, recursive = True):
        #     shutil.rmtree (x)
        
        # create new folders to save the images from the agents
        for i in range(self.no_robot):
            folder_name = "agent_" + str(i+1)
            folder_path = __file__ + "/" + folder_name
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
        
        # create folder to store the top view images
        folder_name = "top_view"
        folder_path = __file__ + "/" + folder_name
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        img_counter = 0
        self.task_over = False
        while not self.task_over:
            print('exec')
            if len(self.action_queue) > 0:
                act = self.action_queue.pop(0)
                self.execute_action(act,img_counter)
                time.sleep(0.5) 
            else:
                self.task_over = True
                print("All tasks completed!", img_counter)
                break
            img_counter += 1    

        # input()
