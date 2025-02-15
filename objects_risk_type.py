import json

# AI2-THOR内置物体及风险类型映射
ai2thor_objects = [
    {"objectType": "Knife", "risk_category": "sharp", "risk_description": "Potential for cutting injuries"},
    {"objectType": "Fork", "risk_category": "sharp", "risk_description": "Potential for piercing injuries"},
    {"objectType": "Scissors", "risk_category": "sharp", "risk_description": "Potential for cutting injuries"},
    {"objectType": "StoveBurner", "risk_category": "thermal", "risk_description": "High heat can cause burns"},
    {"objectType": "Microwave", "risk_category": "thermal", "risk_description": "High heat can cause burns"},
    {"objectType": "Toaster", "risk_category": "thermal", "risk_description": "Hot surfaces can cause burns"},
    {"objectType": "SprayBottle", "risk_category": "chemical", "risk_description": "Potential for chemical exposure"},
    {"objectType": "Bottle", "risk_category": "chemical", "risk_description": "May contain hazardous liquids"},
    {"objectType": "Bathtub", "risk_category": "water", "risk_description": "Potential for drowning"},
    {"objectType": "SinkBasin", "risk_category": "water", "risk_description": "Potential for drowning"},
    {"objectType": "Stairs", "risk_category": "fall", "risk_description": "Potential fall risk"},
    {"objectType": "BalconyEdge", "risk_category": "fall", "risk_description": "Potential fall risk"},
]

# 自定义物体（AI2-THOR中未内置）
custom_objects = [
    {"objectType": "PowerOutlet", "risk_category": "electrical", "risk_description": "Potential for electric shock"},
    {"objectType": "PowerStrip", "risk_category": "electrical", "risk_description": "Potential for electric shock"},
    {"objectType": "CleaningBottle", "risk_category": "chemical", "risk_description": "May contain toxic cleaning agents"},
    {"objectType": "SpaceHeater", "risk_category": "thermal", "risk_description": "High heat surface may cause burns"},
    {"objectType": "BabyCribEdge", "risk_category": "fall", "risk_description": "Risk of falling from crib edge"},
    {"objectType": "WaterBucket", "risk_category": "water", "risk_description": "Potential drowning hazard"},
]

# 添加风险人物（Human Entities），并设定优先级
human_entities = [
    {"objectType": "Baby", "risk_category": "human", "priority": 1, "risk_description": "Vulnerable individual requiring highest safety priority"},
    {"objectType": "Adult", "risk_category": "human", "priority": 2, "risk_description": "Primary household resident with standard safety priority"},
    {"objectType": "Pet", "risk_category": "animal", "priority": 3, "risk_description": "Non-human companion with lower safety priority"}
]

# 合并为完整列表
risk_objects = {
    "ai2thor_objects": ai2thor_objects,
    "custom_objects": custom_objects,
    "human_entities": human_entities
}

# 保存为JSON文件
with open("risk_types.json", "w") as f:
    json.dump(risk_objects, f, indent=4)

print("✅ 风险类型映射文件 `risk_types.json` 生成完成！")
