from ai2thor.controller import Controller
import json

# Define scene categories
kitchens = [f"FloorPlan{i}" for i in range(1, 31)]
living_rooms = [f"FloorPlan{200 + i}" for i in range(1, 31)]
bedrooms = [f"FloorPlan{300 + i}" for i in range(1, 31)]
bathrooms = [f"FloorPlan{400 + i}" for i in range(1, 31)]

# Combine all scenes into a single list
scenes = kitchens + living_rooms + bedrooms + bathrooms

# Dictionary to store all objects and their types in each scene
scene_objects = {}
all_object_types = set()

# Iterate through each scene
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

# Export scene objects to a JSON file
with open("./experiments/data/scene_objects.json", "w") as f:
    json.dump(scene_objects, f, indent=4)

# Export all unique object types to a JSON file
with open("./experiments/data/object_types.json", "w") as f:
    json.dump(sorted(list(all_object_types)), f, indent=4)

print("Export completed!")

# Stop the AI2-THOR controller
controller.stop()
