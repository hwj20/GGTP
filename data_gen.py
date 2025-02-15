import json
from openai import OpenAI
import os

# 读取 ai2thor_object_types.json 和 human_entities.json
with open("./data/ai2thor_object_types.json", "r") as f:
    object_types = json.load(f)

with open("./data/human_entities.json", "r") as f:
    human_entities = json.load(f)["human_entities"]

# 初始化 OpenAI API
api_key = os.getenv("OPENAI_API_KEY")

def generate_prompt(type1, type2):
    return f"""
We have two object types:
- Type 1: {type1}
- Type 2: {type2}
Please evaluate the potential danger between these two types and provide the following information in JSON format:
{
    "danger_level": ["high", "medium", "low"],
    "risk_type": ["sharp", "thermal", "fall", "chemical", "electrical", "water"],
    "llm_reason": "Explanation here"
}
"""

danger_info_list = []
object_types.append([human_entity for human_entity in human_entities])

# 遍历所有类型和人类实体进行两两匹配
for obj_type1 in object_types:
    for obj_type2 in object_types:
        type1 = obj_type1
        type2 = obj_type2
        if type1 == type2:
            continue
        prompt = generate_prompt(type1, type2)
        client = OpenAI(
            api_key=api_key,
        )
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a home safety expert."},
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )        
        result = str(completion.choices[0].message.content)
        print(result)
        input()

        danger_info = {
            "danger_info": {
                "type1": type1,
                "type2": type2,
                "danger_level": None,
                "risk_type": [],
                "llm_reason": result
            }
        }

        # # 根据提供的json解析结果
        # if "high" in result.lower():
        #     danger_info["danger_info"]["danger_level"] = "high"
        # elif "medium" in result.lower():
        #     danger_info["danger_info"]["danger_level"] = "medium"
        # else:
        #     danger_info["danger_info"]["danger_level"] = "low"

        # for risk_type in ["sharp", "thermal", "fall", "chemical", "electrical", "water"]:
        #     if risk_type in result.lower():
        #         danger_info["danger_info"]["risk_type"].append(risk_type)

        danger_info_list.append(danger_info)

# 将标记的数据保存到 JSON 文件
with open("danger_info.json", "w") as f:
    json.dump(danger_info_list, f, indent=4)

print("Danger information saved successfully!")