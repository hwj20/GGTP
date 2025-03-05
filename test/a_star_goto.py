  
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



  # GoToObject: Navigate agent to a target object
def GoToObject(controller, robot, dest_obj, reachable_positions): # Note dest_obj:objectName 
        robot_name = robot['name']
        agent_id = int(robot_name[-1]) - 1
        print(f"Going to {dest_obj} (Agent {agent_id})")

        # Ignore if the destination is "Baby" (simulation logic)
        if "Baby" in dest_obj:
            return

        # Retrieve object metadata
        objects_metadata = controller.last_event.metadata["objects"]
        objs = {obj["name"]: obj for obj in objects_metadata}

        if dest_obj not in objs:
            raise Exception(f"Object '{dest_obj}' not found!")
        
        metadata = controller.last_event.events[agent_id].metadata
        robot_location = metadata["agent"]["position"]
        robot_rotation = metadata["agent"]["rotation"]["y"]

        dest_obj_data = objs[dest_obj]
        dest_obj_pos = dest_obj_data["position"]
        closest_goal = min(reachable_positions, key=lambda p: distance(p, dest_obj_pos))

        # Teleport along path
        path = a_star_planning(robot_location, closest_goal, reachable_positions)

        for i, waypoint in enumerate(path):
            print(f"Teleporting {i+1}/{len(path)}: {waypoint}")

            action_queue.append({
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

        action_queue.append({
            "action": "RotateRight", # if rot_angle > 0 else "RotateLeft",
            "degrees": abs(rot_angle),
            "agent_id": agent_id
        })
