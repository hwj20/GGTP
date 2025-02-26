from graph import *
from task_agent import*
from safe_gat import *
from execute_agent import *
from model.graphormer import Graphormer  # 替换 GAT 为 Graphormer
from safe_gat import SafeGAT

kitchen_graph = build_kitchen_graph()

kitchen_data = graph_to_torch_data(kitchen_graph)
# 训练 Safe-GAT
# model = SafeGAT(in_channels=4, out_channels=2)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# for epoch in range(200):
#     optimizer.zero_grad()
#     out = model(kitchen_data.x, kitchen_data.edge_index)
#     loss = F.mse_loss(out, torch.rand(len(kitchen_graph.nodes), 2))  # 假设性损失计算
#     loss.backward()
#     optimizer.step()

# print("✅ Safe-GAT 训练完成！")

# 训练 Graphormer
model = Graphormer(in_channels=4, out_channels=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(200):
    optimizer.zero_grad()
    out = model(kitchen_data.x, kitchen_data.edge_index)
    loss = F.mse_loss(out, torch.rand(len(kitchen_graph.nodes), 2))  # 假设性损失计算
    loss.backward()
    optimizer.step()

print("✅ Graphormer 训练完成！")


# 计算 Graphormer 输出的注意力
safety_info = generate_safety_recommendations(model(kitchen_data.x, kitchen_data.edge_index), kitchen_graph)
print(safety_info)

# 测试任务生成
task_sequence = generate_task_sequence("Cook dinner", safety_info)
print("📌 LLM 生成的安全任务序列：", task_sequence)

# 解析 JSON 任务数据
import json
task_sequence_parsed = json.loads(task_sequence)

# 机器人执行任务
execute_task_sequence(task_sequence_parsed)
