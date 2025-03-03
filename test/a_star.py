import ai2thor.controller
import heapq
import numpy as np

# 初始化AI2-THOR环境
controller = ai2thor.controller.Controller(
    scene="FloorPlan1",  # 选择一个场景
    gridSize=0.25,       # 设置移动步长
    renderDepthImage=False,
    renderInstanceSegmentation=False
)

def a_star_search(start, goal, grid_map):
    """ A* 路径规划 """
    def heuristic(a, b):
        return np.linalg.norm(np.array(a) - np.array(b))
    
    frontier = []
    heapq.heappush(frontier, (0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}
    
    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            break

        for next_pos in grid_map.get(current, []):
            new_cost = cost_so_far[current] + heuristic(current, next_pos)
            if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                cost_so_far[next_pos] = new_cost
                priority = new_cost + heuristic(goal, next_pos)
                heapq.heappush(frontier, (priority, next_pos))
                came_from[next_pos] = current

    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = came_from.get(node, None)
    return path[::-1]

def move_to_object(controller, target_object_name):
    """ 让机器人跑到物体附近 """
    
    # 1. 获取所有物体的元数据
    objects_metadata = controller.last_event.metadata["objects"]
    
    # 2. 查找目标物体
    target_obj = next((obj for obj in objects_metadata if obj["name"] == target_object_name), None)
    
    if not target_obj:
        raise ValueError(f"Object '{target_object_name}' not found!")

    target_obj_id = target_obj["objectId"]
    
    # 3. 获取目标物体的位置
    event = controller.step(action="GetObjectMetadata", objectIds=[target_obj_id])
    obj_data = event.metadata["objects"][0]  # 目标物品数据
    target_pos = obj_data["position"]

    # 4. 获取可达位置
    event = controller.step(action="GetReachablePositions")
    reachable_positions = event.metadata["actionReturn"]

    # 5. 计算最近的可达点（防止机器人撞到物体）
    closest_pos = min(
        reachable_positions,
        key=lambda p: np.linalg.norm(np.array([p["x"], p["z"]]) - np.array([target_pos["x"], target_pos["z"]]))
    )

    # 6. 运行 A* 规划路径
    path = a_star_search((0, 0), (closest_pos["x"], closest_pos["z"]), reachable_positions)

    # 7. 沿着路径执行 MoveAhead
    for step in path:
        controller.step(action="Teleport", position={"x": step[0], "y": 0.9, "z": step[1]})
        controller.step(action="MoveAhead")

    print(f"机器人到达 {target_object_name} 附近！")

# 让机器人跑到目标物品 "StoveBurner_a0b460e5" 附近
move_to_object(controller, "StoveBurner_a0b460e5")
