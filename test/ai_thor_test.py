'''
The script is to verify ai2-THOR environment
'''
# import time
# import random
# from ai2thor.controller import Controller

# # 初始化 AI2-THOR 环境
# c = Controller(width=1000, height=1000)

# # 重置场景并初始化多个机器人
# print("Initializing AI2-THOR environment...")
# c.reset("FloorPlan1")
# time.sleep(5)  # 等待环境完全加载

# # 初始化两个机器人
# c.step(action="Initialize", agentCount=2)

# # 获取可达位置
# reachable_positions = c.step(action="GetReachablePositions").metadata["actionReturn"]

# # 打印可达位置检查
# print(f"Reachable positions: {reachable_positions[:5]}")

# # 为两个机器人随机选择可达位置
# for i in range(2):
#     init_pos = random.choice(reachable_positions)
#     print(f"Teleporting agent {i} to: {init_pos}")
#     teleport_action = dict(action="Teleport", position=init_pos, agentId=i)
#     result = c.step(teleport_action)
#     if not result.metadata["lastActionSuccess"]:
#         print(f"⚠️ Teleport failed for agent {i}: {result.metadata['errorMessage']}")
#     else:
#         print(f"✅ Agent {i} teleported successfully!")

# print("AI2-THOR environment initialized successfully!")

from ai2thor.controller import Controller
import json

# 初始化AI2-THOR控制器
controller = Controller(scene="FloorPlan1", width=800, height=600)

# 获取当前场景的物体位置信息
def get_all_objects_positions(controller):
    # 触发事件
    event = controller.step(action="Pass")
    object_positions = []

    # 遍历所有物体
    for obj in event.metadata['objects']:
        obj_info = {
            "name": obj['name'],
            "type": obj['objectType'],
            "position": obj['position'],
            "rotation": obj['rotation'],
            "is_receptacle": obj.get('receptacle', False),
            "is_movable": obj.get('moveable', False),
            "parent_receptacle": obj.get('parentReceptacle', None)
        }
        object_positions.append(obj_info)

    # 保存为JSON文件
    with open("floorplan1_object_positions.json", "w") as f:
        json.dump(object_positions, f, indent=4)

    print(f"✅ 已输出 FloorPlan1 的物体位置至 'floorplan1_object_positions.json'！")

    # 控制台预览前10个物体
    for obj in object_positions[:10]:
        print(f"{obj['type']} - Pos: {obj['position']} - Rot: {obj['rotation']}")

# 调用函数
get_all_objects_positions(controller)

