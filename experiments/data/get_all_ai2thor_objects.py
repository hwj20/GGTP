from ai2thor.controller import Controller
import json

# 定义场景列表
kitchens = [f"FloorPlan{i}" for i in range(1, 31)]
living_rooms = [f"FloorPlan{200 + i}" for i in range(1, 31)]
bedrooms = [f"FloorPlan{300 + i}" for i in range(1, 31)]
bathrooms = [f"FloorPlan{400 + i}" for i in range(1, 31)]

scenes = kitchens + living_rooms + bedrooms + bathrooms

# 用于存储所有对象及类型
scene_objects = {}
all_object_types = set()

# 遍历每个场景
controller = Controller()
for scene in scenes:
    controller.reset(scene)
    controller.step(action="Initialize", gridSize=0.25)

    objects = controller.last_event.metadata["objects"]
    scene_objects[scene] = {}
    for obj in objects:
        object_type = obj["objectType"]
        if object_type not in scene_objects[scene]:
            scene_objects[scene][object_type] = []
        scene_objects[scene][object_type].append(obj)
        all_object_types.add(object_type)

# 导出场景对象到JSON文件
with open("./data/scene_objects.json", "w") as f:
    json.dump(scene_objects, f, indent=4)

# 导出所有唯一对象类型到JSON文件
with open("./data/object_types.json", "w") as f:
    json.dump(sorted(list(all_object_types)), f, indent=4)

print("导出完成！")

# 关闭控制器
controller.stop()
