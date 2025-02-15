
from ai2thor.controller import Controller
import random, json, math
from collections import defaultdict

dangers = []
with open("./danger_info.json") as f:
    dangers = json.load(f)
danger_map = defaultdict(dict)
for i in range(len(dangers)):
    danger_info = dangers[i]
    danger_map[danger_info['type1']][danger_info['type2']] = i
    danger_map[danger_info['type2']][danger_info['type1']] = i

# 初始化控制器
controller = Controller(scene='FloorPlan1')
controller.step(action="Initialize", gridSize=0.25)

# 获取场景中的物体
objects = controller.last_event.metadata['objects']

# 随机放置人物和宠物
human_entities = [
    {"type": "Baby", "position": [random.uniform(0, 5), 0, random.uniform(0, 5)]},
    {"type": "Baby", "position": [random.uniform(0, 5), 0, random.uniform(0, 5)]},
    {"type": "Adult", "position": [random.uniform(0, 5), 0, random.uniform(0, 5)]},
    {"type": "Pet", "position": [random.uniform(0, 5), 0, random.uniform(0, 5)]}
]

# 构建节点和边
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

threshold = 0.5
# 计算边并标注危险性
for i, node1 in enumerate(nodes):
    for j, node2 in enumerate(nodes):
        if i >= j:
            continue

        # 计算欧氏距离
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

        # 从 danger_info 里拿，计算总的得分，具体是风险乘以距离系数

        v_map = {'high':1.0,'medium':0.5, 'low':0.25}
        risk_level =  dangers[danger_map[node1['node_type']][node2['node_type']]]['danger_level']
        risk_type =  dangers[danger_map[node1['node_type']][node2['node_type']]]['risk_type']
        danger_score = v_map[risk_level]
        spatio_score = 1/dist # 我也不知道这个合不合理
        label = danger_score*spatio_score

        # 构建边
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

# 保存数据为 JSON
graph_data = {"nodes": nodes, "edges": edges}
with open("graph_data.json", "w") as f:
    json.dump(graph_data, f, indent=4)

print("随机数据生成完成！")
