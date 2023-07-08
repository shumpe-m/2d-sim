#!/usr/bin/env python3
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import torch

from utils.image import  get_area_of_interest_new
from models.models import GraspModel, PlaceModel, MergeModel
from inference.inference import Inference

episode = 49

inf = Inference()

size_input = (480, 752)
size_memory_scale = 4
size_cropped = (250, 250)
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

# print(inf.pose_from_index(np.array([8,40,35,0]), (16, 40, 40, 1)))

# device = "cuda" if torch.cuda.is_available() else "cpu"
# grasp_model = GraspModel(1).float().to(device)
# cptfile = './data/checkpoints/grasp_model.cpt'
# cpt = torch.load(cptfile)
# grasp_model.load_state_dict(cpt['grasp_model_state_dict'])
# grasp_model.eval()


# # image = inf.get_images(image)
# x = np.array(image, dtype=np.float32)
# x = torch.tensor(x).to(device)
# x = torch.reshape(x, (-1,1,image.shape[0],image.shape[1]))
# # for i in range(image.shape[0]): 
# plt.show(block=False)
# plt.gca().axis("off")
# plt.imshow(image, cmap='gray')
# plt.pause(0.7)
# plt.clf()
# plt.close()
# print(x.shape)
# # x = torch.cat([x,x,x,x], dim=0)
# _, reward = grasp_model((x))
# print(torch.max(reward))
# print(torch.min(reward))