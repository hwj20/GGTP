import numpy as np
import math

def calculate_rotate_angle(start, goal, robot_rotation):
    """
    计算机器人从当前朝向，旋转到目标点所需的角度
    :param start: 机器人当前位置 {'x': float, 'z': float}
    :param goal: 目标位置 {'x': float, 'z': float}
    :param robot_rotation: 机器人当前朝向（0-360°）
    :return: 机器人需要旋转的角度（正数 = 顺时针，负数 = 逆时针）
    """
    # 1️⃣ 计算机器人 → 目标的向量
    robot_object_vec = np.array([goal["x"] - start["x"], goal["z"] - start["z"]])

    # 2️⃣ 计算机器人当前朝向向量
    robot_rad = np.deg2rad(robot_rotation)  # 角度转弧度
    robot_dir = np.array([np.cos(robot_rad), np.sin(robot_rad)])  # 机器人前进方向的单位向量

    # 3️⃣ 计算目标向量的角度（atan2 自动处理象限）
    goal_angle = np.degrees(np.arctan2(robot_object_vec[1], robot_object_vec[0]))

    # 4️⃣ 计算旋转角度（归一化到 0-360°）
    goal_angle = (goal_angle + 360) % 360
    rotation_diff = (goal_angle - robot_rotation + 360) % 360

    # 5️⃣ 确定旋转方向（顺时针 or 逆时针）
    if rotation_diff > 180:
        rotation_diff -= 360  # 逆时针旋转（负角度）

    return rotation_diff  # 旋转角度（正数 = 顺时针，负数 = 逆时针）

# 测试一下：
start_pos = {"x": 0, "z": 0}
goal_pos = {"x": 1, "z": 1}
robot_rot = 45  # 机器人当前朝向（单位：度）

rotate_angle = calculate_rotate_angle(start_pos, goal_pos, robot_rot)
print(f"机器人需要旋转 {rotate_angle:.2f}° 才能对准目标！")

