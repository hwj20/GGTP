import json
data = []
res = []
with open("raw_danger_info.json", "r") as f:
    data = json.load(f)
    for item in data:
        item = item['danger_info']
        result = json.loads(item['llm_reason'])
        danger_info = item.copy()
        danger_info['danger_level'] = result['danger_level']
        danger_info['risk_type'] = result['risk_type']
        danger_info['llm_reason'] = result['llm_reason']
        res.append(danger_info)

        # for risk_type in ["sharp", "thermal", "fall", "chemical", "electrical", "water"]:
        #     if risk_type in result.lower():
        #         danger_info["danger_info"]["risk_type"].append(risk_type)

with open('danger_info.json',"w") as f:
    json.dump(res, f,indent=4)
