#!/usr/bin/env python3
import argparse
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import torch

from utils.image import  get_area_of_interest_new
from models.models import GraspModel, PlaceModel, MergeModel
from inference.inference import Inference

parser = argparse.ArgumentParser(description='img test')
parser.add_argument('-e', '--episode', action='store', type=int, default=1)
args = parser.parse_args()

episode = args.episode


# inf = Inference()

size_input = (480, 752)
size_memory_scale = 4
size_cropped = (110, 110)
size_result = (32, 32)
size_cropped_area = (size_cropped[0] // size_memory_scale, size_cropped[1] // size_memory_scale)
image = cv2.imread("./data/img/depth_grasp"+str(episode)+".png", cv2.IMREAD_UNCHANGED)
# image = cv2.imread("./data/img/depth_place_a"+str(episode)+".png", cv2.IMREAD_UNCHANGED)
# image = cv2.imread("./data/goal/cir_goal1.png", cv2.IMREAD_UNCHANGED)
imagea = cv2.resize(image, (size_input[0] // size_memory_scale, size_input[1] // size_memory_scale))


# for i in range(imagea.shape[0]): 
#    plt.show(block=False)
#    plt.gca().axis("off")
#    plt.imshow(imagea[i], cmap='gray')
#    plt.pause(0.3)
#    plt.clf()
#    plt.close()

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
print(pose[2] * 180 / 3.1415)


plt.show(block=False)
plt.gca().axis("off")
plt.imshow(area, cmap='gray')
plt.pause(2.2)
plt.clf()
plt.close()


#################################

# size_input = (480, 752)
# size_memory_scale = 8
# size_cropped = (220, 220)
# size_result = (32, 32)
# size_cropped_area = (size_cropped[0] // size_memory_scale, size_cropped[1] // size_memory_scale)
# image = cv2.imread("./data/img/depth_grasp"+str(episode)+".png", cv2.IMREAD_UNCHANGED)
# # image = cv2.imread("./data/goal/cir_goal1.png", cv2.IMREAD_UNCHANGED)
# imagea = cv2.resize(image, (size_input[0] // size_memory_scale, size_input[1] // size_memory_scale))


# # for i in range(imagea.shape[0]): 
# #    plt.show(block=False)
# #    plt.gca().axis("off")
# #    plt.imshow(imagea[i], cmap='gray')
# #    plt.pause(0.3)
# #    plt.clf()
# #    plt.close()

# # dir = "./data/obj_info/obj_info.json"
# # with open(dir, mode="rt", encoding="utf-8") as f:
# #             obj_infos = json.load(f)
# # pose = obj_infos["0"]["center_psoe"]
# # pose = np.array([221, 198, 0])
# # pose[0] = 240
# # pose[1] = 752/2
# # print(pose)

# with open("./data/datasets/grasp_datasets.json", mode="rt", encoding="utf-8") as f:
#     all_data = json.load(f)
# pose = all_data[str(episode)]["grasp"]["pose"]
# area = get_area_of_interest_new(
#     imagea,
#     pose,
#     size_cropped=size_cropped_area,
#     size_result=size_result,
#     size_memory_scale = size_memory_scale,
# )


# plt.show(block=False)
# plt.gca().axis("off")
# plt.imshow(area, cmap='gray')
# plt.pause(2)
# plt.clf()
# plt.close()

############################



# device = "cuda" if torch.cuda.is_available() else "cpu"
# grasp_model = GraspModel(1).float().to(device)
# cptfile = './data/checkpoints/grasp_model.cpt'
# cpt = torch.load(cptfile)
# grasp_model.load_state_dict(cpt['grasp_model_state_dict'])
# grasp_model.eval()


# image = inf.get_images(image)
# print(image.shape)
# # for i in range(image.shape[0]): 
# #     plt.show(block=False)
# #     plt.gca().axis("off")
# #     plt.imshow(image[i], cmap='gray')
# #     plt.pause(0.7)
# #     plt.clf()
# #     plt.close()
# x = np.array(image, dtype=np.float32)
# x = torch.tensor(x).to(device)
# x = torch.reshape(x, (-1,1,image.shape[1],image.shape[2]))
#     # print(x.shape)
# # x = torch.cat([x,x,x,x], dim=0)
# _, reward = grasp_model((x))
# print(reward.shape)
# reward = reward[:,0]
# print(reward.shape)
# reward = reward.unsqueeze(dim=1)
# reward = reward.to('cpu').detach().numpy().copy()
# p_index = np.unravel_index(np.argmax(reward), reward.shape)
# print(p_index)
# print(inf.pose_from_index(p_index, reward.shape, resolution_factor=1.0))