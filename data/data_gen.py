from ai2thor.controller import Controller
import random, json, math, os
from collections import defaultdict

# 读取危险信息
with open("./data/danger_info.json") as f:
    dangers = json.load(f)
danger_map = defaultdict(dict)
for i in range(len(dangers)):
    danger_info = dangers[i]
    danger_map[danger_info['type1']][danger_info['type2']] = i
    danger_map[danger_info['type2']][danger_info['type1']] = i

# 生成所有 AI2-THOR 正确的场景列表(120个数据集)
kitchens = [f"FloorPlan{i}" for i in range(1, 31)]
living_rooms = [f"FloorPlan{200 + i}" for i in range(1, 31)]
bedrooms = [f"FloorPlan{300 + i}" for i in range(1, 31)]
bathrooms = [f"FloorPlan{400 + i}" for i in range(1, 31)]

# 合并所有场景
scenes = kitchens + living_rooms + bedrooms + bathrooms

random.shuffle(scenes)
train_scenes = scenes[:80]
val_scenes = scenes[80:100]
test_scenes = scenes[100:]

# 数据集存储
dataset = {"train": [], "val": [], "test": []}

def generate_scene(scene_name):
    controller = None
    try:
        controller = Controller(scene=scene_name)
        controller.step(action="Initialize", gridSize=0.25)
        objects = controller.last_event.metadata['objects']
        
        nodes = []
        edges = []
        node_id = 0
        
        # 添加物体节点
        for obj in objects:
            nodes.append({
                "node_id": node_id,
                "node_type": obj['objectType'],
                "features": {
                    "temperature": obj.get("temperature", 20),
                    "energy_source": obj.get("energySource", "none"),
                    "position": list(obj["position"].values())
                }
            })
            node_id += 1
        
        # 确保至少 50% 场景有危险边
        insert_danger = random.random() > 0.5
        human_entities = []
        
        if insert_danger:
            high_risk_objects = [obj for obj in objects if obj["objectType"] in danger_map]
            if high_risk_objects:
                dangerous_obj = random.choice(high_risk_objects)
                dangerous_pos = list(dangerous_obj["position"].values())
                # 在危险物附近随机放一个人
                human_entities.append({
                    "type": "Baby",  # 高风险实验
                    "position": [dangerous_pos[0] + random.uniform(-0.5, 0.5), 0, dangerous_pos[2] + random.uniform(-0.5, 0.5)]
                })
        
        # 添加随机人物
        human_entities += [
            {"type": "Adult", "position": [random.uniform(0, 5), 0, random.uniform(0, 5)]},
            {"type": "Pet", "position": [random.uniform(0, 5), 0, random.uniform(0, 5)]}
        ]
        
        # 添加人物和宠物节点
        for entity in human_entities:
            nodes.append({
                "node_id": node_id,
                "node_type": entity["type"],
                "features": {
                    "temperature": 36.5,
                    "energy_source": "none",
                    "position": entity["position"]
                }
            })
            node_id += 1
        
        # 计算边并标注危险性
        threshold = 0.5
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes):
                if i >= j:
                    continue
                dist = math.dist(node1['features']['position'], node2['features']['position'])
                
                if node2['node_type'] not in danger_map[node1['node_type']]:
                    edges.append({
                        "edge_id": len(edges),
                        "node1_id": i,
                        "node2_id": j,
                        "distance": round(dist, 2),
                        "edge_type": 0,
                        "risk_level": 'None',
                        "risk_type": [],
                        "attention_bias": 0
                    })
                    continue
                try:
                    v_map = {'high':1.0,'medium':0.5, 'low':0.25}
                    risk_level = dangers[danger_map[node1['node_type']][node2['node_type']]]['danger_level']
                    risk_type = dangers[danger_map[node1['node_type']][node2['node_type']]]['risk_type']
                    danger_score = v_map[risk_level]
                    spatio_score = 1 / dist if dist > 0 else 1.0
                    label = danger_score * spatio_score
                    
                    edges.append({
                        "edge_id": len(edges),
                        "node1_id": i,
                        "node2_id": j,
                        "distance": round(dist, 2),
                        "edge_type": 1 if label > threshold else 0,
                        "risk_level": risk_level,
                        "risk_type": risk_type,
                        "attention_bias": label
                    })
                except Exception as e:
                    print(f"⚠️ Error in scene {scene_name} between nodes {i} and {j}: {str(e)}")
    
        return {"scene": scene_name, "nodes": nodes, "edges": edges}
    
    finally:
        if controller:
            controller.stop()  # 释放 AI2-THOR 进程

# 生成所有场景数据
def generate_dataset():
    os.system("pkill -f ai2thor")  # 运行前清理旧进程
    for scene in train_scenes:
        dataset["train"].append(generate_scene(scene))
    for scene in val_scenes:
        dataset["val"].append(generate_scene(scene))
    for scene in test_scenes:
        dataset["test"].append(generate_scene(scene))
    
    # 保存 JSON
    with open("graph_dataset.json", "w") as f:
        json.dump(dataset, f, indent=4)
    print("随机数据生成完成！")

generate_dataset()
