import ai2thor.controller
import cv2
import numpy as np

# 初始化 AI2-THOR
controller = ai2thor.controller.Controller(
    scene="FloorPlan1",  # 选择一个厨房场景
    gridSize=0.25,
    width=640,
    height=640
)

# **调整 Top View 角度，确保能看到整个房间**
top_view_params = {
    "rotation": {"x": 90, "y": 0, "z": 0},  # 角度从 90° 调整到 45°，避免只看地面
    "position": {"x": 0, "y": 2.5, "z": 0},  # 提高相机高度，确保整个房间能看到
    "fieldOfView": 100  # 调整视角宽度，让房间内容更多
}

# **添加俯视相机**
# controller.step(
#     action="AddThirdPartyCamera",
#     position=top_view_params["position"],
#     rotation=top_view_params["rotation"],
#     fieldOfView=top_view_params["fieldOfView"]
# )
event = controller.step(action="GetMapViewCameraProperties")
controller.step(action="AddThirdPartyCamera", **event.metadata["actionReturn"])

# **开始显示 Top View 画面**
while True:
    event = controller.step(action="Pass")
    top_view_img = event.third_party_camera_frames[-1]  # 取 top-view 相机画面
    # spawnable_objs = controller.step(action="GetSpawnableObjects").metadata.get("actionReturn", [])
    # print("可生成的物体:", spawnable_objs)


    # **转换格式（AI2-THOR 默认 RGB，OpenCV 需要 BGR）**
    img_bgr = cv2.cvtColor(top_view_img, cv2.COLOR_RGB2BGR)
    top_view_rgb = cv2.cvtColor(controller.last_event.events[0].third_party_camera_frames[-1], cv2.COLOR_RGB2BGR)
    cv2.imshow('Top View', top_view_rgb)

    # **显示画面**
    cv2.imshow("Top View Camera", img_bgr)

    # 按 "q" 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# **关闭窗口**
cv2.destroyAllWindows()
controller.stop()
