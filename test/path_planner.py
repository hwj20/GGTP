import random
import cv2
import time
import math
import numpy as np
from ai2thor.controller import Controller
__file__ = "./testing/"
image_counter = 0
# Initialize AI2-THOR
controller = Controller(
    scene="FloorPlan212",
    visibilityDistance=1.5,
    gridSize=0.25,
    snapToGrid=True
)
# add a top view camera
event = controller.step(action="GetMapViewCameraProperties")
event = controller.step(action="AddThirdPartyCamera", **event.metadata["actionReturn"])

# Get all reachable positions & set initial agent position
event = controller.step(action="GetReachablePositions")
positions = event.metadata["actionReturn"]
initial_position = random.choice(positions)

# Place agent at the selected initial position
controller.step(action="TeleportFull", position=initial_position, rotation=0, horizon=0,standing=True)

# Ensure agent exists
agent_position = controller.last_event.metadata["agent"]["position"]
print(f"Agent initialized at {agent_position}")


# Find the shortest path to a target object
target_object = "Television"
event = controller.step(
    action="GetShortestPath",
    objectType=target_object,
    position=initial_position,
    allowedError=0.05
)

if not event.metadata["lastActionSuccess"]:
    raise ValueError(f"No valid path found to {target_object}!")

shortest_path = event.metadata["actionReturn"]["corners"]
print(f"Path to {target_object} retrieved! {len(shortest_path)} waypoints.")
object_metadata = next(obj for obj in controller.last_event.metadata["objects"] if obj["objectType"] == target_object)
shortest_path.append(object_metadata["position"])

# Function to compute Euclidean distance
def distance(a, b):
    return math.sqrt((a["x"] - b["x"]) ** 2 + (a["z"] - b["z"]) ** 2)

# Function to rotate the agent to face a target point
def rotate_to_target(controller, next_pos):
    global image_counter
    agent_pos = controller.last_event.metadata["agent"]["position"]
    agent_rot = controller.last_event.metadata["agent"]["rotation"]["y"]

    dx = next_pos["x"] - agent_pos["x"]
    dz = next_pos["z"] - agent_pos["z"]

    # 计算应该旋转的目标角度
    correct_angle = math.degrees(math.atan2(dz, dx))
    correct_angle = (correct_angle + 360) % 360  # 归一化到 [0, 360]

    print(f"🔍 DEBUG: dx={dx}, dz={dz}, atan2(dz, dx)={math.atan2(dz, dx)} rad, correct_angle={correct_angle}°")

    # 计算旋转方向（顺时针 or 逆时针）
    rot_angle = (correct_angle - agent_rot + 360) % 360
    if rot_angle > 180:
        controller.step(action="RotateLeft", degrees=360 - rot_angle, agentId = 0)
        print(f"Rotating Left by {360 - rot_angle}°")
    else:
        controller.step(action="RotateRight", degrees=rot_angle, agentId = 0)
        print(f"Rotating Right by {rot_angle}°")
    save_frame(image_counter)
    image_counter += 1

def get_nearest_reachable_position(controller, waypoint):
    """ 获取 `waypoint` 附近最近的可达点 """
    reachable_positions = controller.step(action="GetReachablePositions").metadata["actionReturn"]

    # 计算 `reachablePositions` 中距离 `waypoint` 最近的点
    nearest_pos = min(reachable_positions, key=lambda pos: distance(pos, waypoint))

    return nearest_pos


def save_frame(img_counter):
    tag = "planner_"
    top_view = controller.last_event.events[0].third_party_camera_frames[-1]
    top_view_bgr = cv2.cvtColor(top_view, cv2.COLOR_RGB2BGR)
    cv2.imshow('Top View', top_view)
    f_name = __file__+'//'+tag+ "top_view/img_" + str(img_counter).zfill(5) + ".png"
    cv2.imwrite(f_name, top_view_bgr)

# Function to move the agent step-by-step towards waypoints
def move_agent(controller, path, step_size=0.25, threshold=0.15):
    global image_counter
    """ Move agent smoothly along the shortest path using Teleport, with dynamic path correction. """
    for i, waypoint in enumerate(path):
        print(f"Moving towards waypoint {i + 1}/{len(path)}: {waypoint}")
        nearest_waypoint = get_nearest_reachable_position(controller, waypoint)
        print(f"  Adjusted waypoint to nearest reachable position: {nearest_waypoint}")
        agent_pos = controller.last_event.metadata["agent"]["position"]
        dis = distance(agent_pos,nearest_waypoint)

        if dis <= threshold or distance(agent_pos,waypoint) <= threshold:
            continue 



        # 让 AI 逐步移动
        event = controller.step(action="Teleport", position=nearest_waypoint, agentId=0)
        save_frame(image_counter)
        image_counter += 1

        if not event.metadata["lastActionSuccess"]:
            return
            #     event = controller.step(
            #         action="GetShortestPath",
            #         objectType=target_object,
            #         position=controller.last_event.metadata["agent"]["position"],
            #         allowedError=0.05
            #     )

            #     if not event.metadata["lastActionSuccess"]:
            #         print(f"⚠️ Failed to find a new path to {target_object}. Stopping.")
            #         return

            #     # 更新 `path`，让 AI 继续前进
            #     path = event.metadata["actionReturn"]["corners"]
            #     move_agent(controller,path,step_size,threshold)
            #     return

            # time.sleep(0.1)  # 让 AI 走得更流畅

    print(f"Successfully reached the last waypoint!")

    # Final adjustment: Face the target object
    object_metadata = next(obj for obj in controller.last_event.metadata["objects"] if obj["objectType"] == target_object)
    rotate_to_target(controller, object_metadata["position"])
    print(f"Agent is now facing {target_object}!")

# Move agent along the path smoothly
move_agent(controller, shortest_path)

# Stop AI2-THOR
controller.stop()
