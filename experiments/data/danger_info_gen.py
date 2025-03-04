import json
from openai import OpenAI
import os

# read ai2thor_object_types.json 和 human_entities.json
with open("./experiments/data/ai2thor_object_types.json", "r") as f:
    object_types = json.load(f)

with open("./experiments/data/human_entities.json", "r") as f:
    human_entities = json.load(f)["human_entities"]

# initialize OpenAI API
api_key = os.getenv("OPENAI_API_KEY")

def generate_prompt(type1, type2):
    return f"""
We have two object types:
- Type 1: {type1}
- Type 2: {type2}
Please evaluate the potential danger between these two types and provide the following information in JSON format:
{{
    "danger_level": ["high", "medium", "low"],
    "risk_type": ["physical", "thermal", "fall", "chemical", "electrical", "water"],
    "llm_reason": "Explanation here"
}}
PLEASE respond only with the JSON string without any markdown characters or additional descriptions.
"""

danger_info_list = []
object_types = [human_entity['objectType'] for human_entity in human_entities] + object_types
print(len(object_types))
# visited all objects and human entities
for i in range(len(object_types)):
    for j in range(i+1,len(object_types)):
        type1 = object_types[i]
        type2 = object_types[j]
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

        danger_info = {
            "danger_info": {
                "type1": type1,
                "type2": type2,
                "danger_level": None,
                "risk_type": [],
                "llm_reason": result # raw data; will be processed in process_raw_data.py
            }
        }


        danger_info_list.append(danger_info)

        # save data
        with open("./experiments/data/danger_info.json", "w") as f:
            json.dump(danger_info_list, f, indent=4)

        print(f"Saved entry: {type1} -> {type2}")
        print(danger_info)
print("Danger information saved successfully!")