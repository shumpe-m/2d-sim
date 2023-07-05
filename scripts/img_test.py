#!/usr/bin/env python3
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from utils.image import  get_area_of_interest_new

episode = 198

size_input = (480, 752)
size_memory_scale = 4
size_cropped = (200, 200)
size_result = (32, 32)
size_cropped_area = (size_cropped[0] // size_memory_scale, size_cropped[1] // size_memory_scale)
image = cv2.imread("./data/img/depth_grasp"+str(episode)+".png", cv2.IMREAD_UNCHANGED)
imagea = cv2.resize(image, (size_input[0] // size_memory_scale, size_input[1] // size_memory_scale))
# dir = "./data/obj_info/obj_info.json"
# with open(dir, mode="rt", encoding="utf-8") as f:
#             obj_infos = json.load(f)
# pose = obj_infos["0"]["center_psoe"]
# pose = np.array([221, 198, 0])
# pose[0] = 240
# pose[1] = 752/2
# print(pose)
with open("./data/datasets/datasets.json", mode="rt", encoding="utf-8") as f:
    all_data = json.load(f)
pose = all_data[str(episode)]["grasp"]["pose"]
area = get_area_of_interest_new(
    imagea,
    pose,
    size_cropped=size_cropped_area,
    size_result=size_result,
    size_memory_scale = size_memory_scale,
)
plt.show(block=False)
plt.gca().axis("off")
plt.imshow(area, cmap='gray')
plt.pause(2)
plt.clf()
plt.close()