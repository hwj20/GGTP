import threading
import time
import cv2
import ai2thor.controller
__file__ = "./testing/"
image_counter = 0
def save_frame(img_counter):
    tag = "planner_"
    top_view = controller.last_event.events[0].third_party_camera_frames[-1]
    top_view_bgr = cv2.cvtColor(top_view, cv2.COLOR_RGB2BGR)
    cv2.imshow('Top View', top_view)
    f_name = __file__+'//'+tag+ "top_view/img_" + str(img_counter).zfill(5) + ".png"
    cv2.imwrite(f_name, top_view_bgr)


import ai2thor.controller
controller = ai2thor.controller.Controller()
# controller.start()

controller.reset('FloorPlan28')
controller.step(dict(action='Initialize', gridSize=0.25))
# add a top view camera
event = controller.step(action="GetMapViewCameraProperties")
event = controller.step(action="AddThirdPartyCamera", **event.metadata["actionReturn"])

controller.step(dict(action='Teleport', x=-2.5, y=0.900998235, z=-3.0))
controller.step(dict(action='LookDown'))
# event = controller.step(dict(action='Rotate', rotation=180))
# In FloorPlan28, the agent should now be looking at a mug
for o in event.metadata['objects']:
    if o['visible'] and o['pickupable'] and o['objectType'] == 'Mug':
        event = controller.step(dict(action='PickupObject', objectId=o['objectId']), raise_for_failure=True)
        mug_object_id = o['objectId']
        break

# the agent now has the Mug in its inventory
# to put it into the Microwave, we need to open the microwave first

event = controller.step(dict(action='LookUp'))
save_frame(image_counter)
image_counter += 1
event = controller.step(dict(action='RotateLeft', degrees=30))
save_frame(image_counter)
image_counter += 1

event = controller.step(dict(action='MoveLeft'))
save_frame(image_counter)
image_counter += 1
event = controller.step(dict(action='MoveLeft'))
save_frame(image_counter)
image_counter += 1
event = controller.step(dict(action='MoveLeft'))
save_frame(image_counter)
image_counter += 1
event = controller.step(dict(action='MoveLeft'))

event = controller.step(dict(action='MoveAhead'))
save_frame(image_counter)
image_counter += 1
event = controller.step(dict(action='MoveAhead'))
save_frame(image_counter)
image_counter += 1
event = controller.step(dict(action='MoveAhead'))
event = controller.step(dict(action='MoveAhead'))
event = controller.step(dict(action='MoveAhead'))
event = controller.step(dict(action='MoveAhead'))

for o in event.metadata['objects']:
    if o['visible'] and o['openable'] and o['objectType'] == 'Microwave':
        event = controller.step(dict(action='OpenObject', objectId=o['objectId']), raise_for_failure=True)
        receptacle_object_id = o['objectId']
        break

event = controller.step(dict(
    action='PutObject',
    receptacleObjectId=receptacle_object_id,
    objectId=mug_object_id), raise_for_failure=True)

# close the microwave
event = controller.step(dict(
    action='CloseObject',
    objectId=receptacle_object_id), raise_for_failure=True)