import ai2thor.controller
import matplotlib.pyplot as plt

controller = ai2thor.controller.Controller(scene="FloorPlan1", gridSize=0.25)

# add a top view camera
event = controller.step(action="GetMapViewCameraProperties")
event = controller.step(action="AddThirdPartyCamera", **event.metadata["actionReturn"])

controller.step(
    action="CreateObject",
    objectType="Sphere",
    position={"x": -1.0, "y": 0.2, "z": 1.5},
    scale={"x": 0.4, "y": 0.4, "z": 0.4},
    color=[255, 182, 193]
)

event = controller.step(action="Pass")
frame = event.frame  

plt.imshow(frame)
plt.axis("off")  
plt.title("Top-Down View of the Scene")  
plt.show()
