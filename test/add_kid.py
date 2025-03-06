import ai2thor.controller
import matplotlib.pyplot as plt

# 启动 AI2-THOR 控制器
controller = ai2thor.controller.Controller(scene="FloorPlan1", gridSize=0.25)

# add a top view camera
event = controller.step(action="GetMapViewCameraProperties")
event = controller.step(action="AddThirdPartyCamera", **event.metadata["actionReturn"])

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
