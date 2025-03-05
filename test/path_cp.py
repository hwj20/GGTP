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


def save_frame(img_counter):
    tag = "planner_"
    top_view = controller.last_event.events[0].third_party_camera_frames[-1]
    top_view_bgr = cv2.cvtColor(top_view, cv2.COLOR_RGB2BGR)
    cv2.imshow('Top View', top_view)
    f_name = __file__+'//'+tag+ "top_view/img_" + str(img_counter).zfill(5) + ".png"
    cv2.imwrite(f_name, top_view_bgr)


def teleport_along_path(controller, reachable_positions, step_size=0.5):
    """ 机器人逐步传送，假装路径规划 """
    global image_counter

    # 1️⃣ 获取目标物体的位置
    object_metadata = next(obj for obj in controller.last_event.metadata["objects"] if obj["objectType"] == target_object)
    target_pos = object_metadata["position"]

    # 2️⃣ 找到目标最近的可达点
    closest_goal = min(reachable_positions, key=lambda p: distance(p, target_pos))
    print(f"🎯 目标最近可达点: {closest_goal}")

    # 3️⃣ 获取当前机器人位置
    agent_pos = controller.last_event.metadata["agent"]["position"]
    current_pos = {"x": agent_pos["x"], "y": agent_pos["y"], "z": agent_pos["z"]}

    # 4️⃣ 计算路径（这里可以用 A* 或者直接取可达点中最短路径）
    path = a_star_planning(current_pos, closest_goal, reachable_positions)

    # 5️⃣ 逐步传送，模拟“智能导航”
    for i, waypoint in enumerate(path):
        print(f"🚀 传送到路径点 {i+1}/{len(path)}: {waypoint}")

        # 传送机器人到 waypoint
        controller.step(
            action="Teleport",
            position=waypoint,
            agentId=0
        )

        # 记录帧，让视频看起来流畅
        save_frame(image_counter)
        image_counter += 1

    # 6️⃣ 旋转机器人面向目标
    rotate_to_target(controller, target_pos)
    print(f"✅ 机器人已顺利抵达 {target_object}，并正对目标！")

def a_star_planning(start, goal, reachable_positions):
    """ 超简化 A* 计算路径（这里只是简单取最短路径点，避免复杂计算）"""
    
    path = [start]
    
    while distance(path[-1], goal) > 0.5:  # 设定步长
        next_pos = min(reachable_positions, key=lambda p: distance(p, goal) + distance(p, path[-1]))
        if next_pos in path:
            break  # 防止死循环
        path.append(next_pos)

    path.append(goal)  # 最后一步直接传送到目标
    return path

print(positions)

# Move agent along the path smoothly
teleport_along_path(controller,positions)

# Stop AI2-THOR
controller.stop()
