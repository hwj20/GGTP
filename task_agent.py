import os
from openai import OpenAI

# 初始化 OpenAI 接口
api_key = os.getenv("OPENAI_API_KEY")


def generate_task_sequence(task_description, action_list, environment_objects):
    
    """
    调用 LLM 根据任务描述和环境信息生成任务序列
    """ 
    response_json = '''
    {
    "action": "<action_name>",  // Available: "GoToObject", "PickupObject", "PutObject", "SwitchOn", "SwitchOff"
    "object_id": "<object_id>"  // object_id in envirnoment
    }
    '''
    prompt = f"""
    Task: {task_description}
    Environment: {environment_objects}

    Generate a step-by-step task sequence in JSON format. Each step should include:
    1. The action to perform. 
    2. The target object for the action, represented as an object with name and coordinates.

    Response json format:{response_json}

    PLEASE respond only with the JSON string without any markdown characters or additional descriptions.
    """

    system_prompt = "You are a smart robot assistant. Your job is to generate a task sequence for the given task and environment."


    try:
        client = OpenAI(
            api_key=api_key,
        )
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        generated_answer = str(completion.choices[0].message.content)
        return generated_answer
    except Exception as e:
        return f"API调用错误: {e}"



# 解析生成的任务序列
def parse_task_sequence(task_sequence_json):
    """
    将 LLM 输出的 JSON 解析为 Python 数据结构
    """
    import json
    try:
        task_sequence = json.loads(task_sequence_json)
        return task_sequence
    except json.JSONDecodeError as e:
        print("Failed to parse task sequence:", e)
        return []


# 测试任务生成
def main():
    # 示例输入
    task_description = "切菜"
    environment_objects = [
        {"name": "fridge", "type": "object", "coordinates": (0.5, 1.2, 0.0), "state": "closed"},
        {"name": "knife", "type": "object", "coordinates": (0.2, 0.8, 0.0), "state": "sharp"},
        {"name": "child", "type": "human", "coordinates": (0.8, 1.5, 0.0), "state": "playing"},
    ]

    # Step 1: 调用 LLM 生成任务序列
    task_sequence_json = generate_task_sequence(task_description, environment_objects)
    print("Generated Task Sequence (JSON):")
    print(task_sequence_json)

    # Step 2: 解析任务序列
    task_sequence = parse_task_sequence(task_sequence_json)
    print("\nParsed Task Sequence:")
    for step in task_sequence:
        print(step)


if __name__ == "__main__":
    main()
