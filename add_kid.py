import ai2thor.controller
import matplotlib.pyplot as plt

# 启动 AI2-THOR 控制器
controller = ai2thor.controller.Controller(scene="FloorPlan1", gridSize=0.25)

# **调整 Agent 位置，确保摄像头从上往下**
controller.step(
    action="TeleportFull",
    x=0, y=2.5, z=0,  
    rotation=dict(x=0, y=0, z=0),
    horizon=90,  # 俯视角度
    standing=True  # ！！！一定要加上 standing=True，不然会报错
)

# **创建小孩球体**
controller.step(
    action="CreateObject",
    objectType="Sphere",
    position={"x": -1.0, "y": 0.2, "z": 1.5},
    scale={"x": 0.4, "y": 0.4, "z": 0.4},
    color=[255, 182, 193]
)

# **获取当前场景截图**
event = controller.step(action="Pass")
frame = event.frame  

# **用 Matplotlib 显示**
plt.imshow(frame)
plt.axis("off")  
plt.title("Top-Down View of the Scene")  
plt.show()
