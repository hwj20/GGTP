def execute_task_sequence(task_sequence):
    for step in task_sequence:
        action, target = step["action"], step.get("target", "")
        print(f"🤖 Executing: {action} {target}")

        # 更新环境状态（模拟）
        if action == "pick_up":
            print(f"✅ {target} is now in robot's hand.")
        elif action == "cut":
            print(f"✅ {target} has been chopped.")
        elif action == "wait":
            print(f"⏳ Waiting for safe conditions...")
