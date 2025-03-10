'''
The script is to verify the AI2-THOR environment
'''
# import time
# import random
# from ai2thor.controller import Controller

# Initialize the AI2-THOR environment
c = Controller(width=1000, height=1000)

# Reset the scene and initialize multiple agents
print("Initializing AI2-THOR environment...")
c.reset("FloorPlan1")
time.sleep(5)  # Wait for the environment to fully load

# Initialize two agents
c.step(action="Initialize", agentCount=2)

# Get all reachable positions
reachable_positions = c.step(action="GetReachablePositions").metadata["actionReturn"]

# Print a subset of reachable positions for verification
print(f"Reachable positions: {reachable_positions[:5]}")

# Randomly teleport two agents to reachable positions
for i in range(2):
    init_pos = random.choice(reachable_positions)
    print(f"Teleporting agent {i} to: {init_pos}")
    teleport_action = dict(action="Teleport", position=init_pos, agentId=i)
    result = c.step(teleport_action)
    if not result.metadata["lastActionSuccess"]:
        print(f"⚠️ Teleport failed for agent {i}: {result.metadata['errorMessage']}")
    else:
        print(f"✅ Agent {i} teleported successfully!")

print("AI2-THOR environment initialized successfully!")

from ai2thor.controller import Controller
import json

# Initialize AI2-THOR controller
controller = Controller(scene="FloorPlan1", width=800, height=600)

# Retrieve the positions of all objects in the current scene
def get_all_objects_positions(controller):
    # Execute an action
    event = controller.step(action="Pass")
    object_positions = []

    # Iterate through all objects
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

    # Save the object data as a JSON file
    with open("floorplan1_object_positions.json", "w") as f:
        json.dump(object_positions, f, indent=4)

    print(f"✅ Object positions for FloorPlan1 saved to 'floorplan1_object_positions.json'.")

    # Preview the first 10 objects in the console
    for obj in object_positions[:10]:
        print(f"{obj['type']} - Pos: {obj['position']} - Rot: {obj['rotation']}")

# Execute the function
get_all_objects_positions(controller)
