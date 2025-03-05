import os
from openai import OpenAI

# Initialize OpenAI key 
api_key = os.getenv("OPENAI_API_KEY")
def generate_task_sequence_safety_prompt_llm(task_description, action_list, environment_objects):
    """
    Invoke the LLM to generate a task sequence based on the task description and environment information
    """ 
    response_json = f'''
    {{
    "action": "<action_name>",  // Available: {action_list}
    "object_id": "<object_id>"  // object_id in environment
    "target_id": "<target_container>" // if the action is "PutObject", then set this var
    }}
    or
    {{
    "action": "Done"    // it must be put at the end of a sequence
    }}
    or
    {{
    "action": "HandleSafetyIssue"
    "object_id": "<object_id>"
    }}
    '''
    prompt = f"""
    Task: {task_description}
    Environment: {environment_objects}
    If a safety issue is detected, first move to the location of the issue and perform 'HandleSafetyIssue' on the affected entity.

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
        return f"An error during API call: {e}"

def generate_task_sequence_llm_only(task_description, action_list, environment_objects):
    """
    Invoke the LLM to generate a task sequence based on the task description and environment information
    """ 
    response_json = f'''
    {{
    "action": "<action_name>",  // Available: {action_list}
    "object_id": "<object_id>"  // object_id in environment
    "target_id": "<target_container>" // if the action is "PutObject", then set this var
    }}
    or
    {{
    "action": "Done"    // it must be put at the end of a sequence
    }}
    or
    {{
    "action": "HandleSafetyIssue"
    "object_id": "<object_id>"
    }}
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
        return f"An error during API call: {e}"



def generate_task_sequence(task_description, action_list, environment_objects, safety_notice):
    
    """
    Invoke the LLM to generate a task sequence based on the task description and environment information
    """ 
    response_json = f'''
    {{
    "action": "<action_name>",  // Available: {action_list}
    "object_id": "<object_id>"  // object_id in environment
    "target_id": "<target_container>" // if the action is "PutObject", then set this var
    }}
    or
    {{
    "action": "Done"    // it must be put at the end of a sequence
    }}
    or
    {{
    "action": "HandleSafetyIssue"
    "object_id": "<object_id>"
    }}
    '''
    prompt = f"""
    Task: {task_description}
    Environment: {environment_objects}
    Current Safety Situation: {safety_notice}.
    If a safety issue is detected, first move to the location of the issue and perform 'HandleSafetyIssue' on the affected entity.

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
        return f"An error during API call: {e}"



# parse generated tasks
def parse_task_sequence(task_sequence_json):
    """
    Parse the JSON output from the LLM into a Python data structure
    """
    import json
    try:
        task_sequence = json.loads(task_sequence_json)
        return task_sequence
    except json.JSONDecodeError as e:
        print("Failed to parse task sequence:", e)
        return []


# testing
def main():
    # sample input
    task_description = "goto fridge"
    environment_objects = [
        {"name": "fridge", "type": "object", "coordinates": (0.5, 1.2, 0.0), "state": "closed"},
        {"name": "knife", "type": "object", "coordinates": (0.2, 0.8, 0.0), "state": "sharp"},
        {"name": "child", "type": "human", "coordinates": (0.8, 1.5, 0.0), "state": "playing"},
    ]

    # Step 1: Call the LLM to generate a task sequence
    task_sequence_json = generate_task_sequence(task_description,action_list=['goto'], environment_objects=environment_objects,safety_notice='safe')
    print("Generated Task Sequence (JSON):")
    print(task_sequence_json)

    # Step 2: parse task sequence
    task_sequence = parse_task_sequence(task_sequence_json)
    print("\nParsed Task Sequence:")
    for step in task_sequence:
        print(step)


if __name__ == "__main__":
    main()
