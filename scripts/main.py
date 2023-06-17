#!/usr/bin/env python3
import os
from subprocess import Popen
import sys

import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
from pathlib import Path

from learning.train import Train
from inference.inference import Inference
from inference.inference_utils import InferenceUtils
from environment import Environment
from utils.param import Mode, SelectionMethod
from action.grasp_decision import GraspDecision

# train = Train()
# train.train()

# src = cv2.imread("./data/img/rgb_test.png", cv2.IMREAD_UNCHANGED)
# # src = cv2.imread("./data/img/depth_grasp0.png", cv2.IMREAD_GRAYSCALE)
# infe = InferenceUtils()
# # dst = cv2.resize(src, (200, 200))
# img = infe.get_images(orig_image = src)

# print(src.dtype)

# plt.show(block=False)
# plt.gca().axis("off")

# plt.imshow(src, cmap='gray')
# plt.pause(2)
# for i in range(16):
#    plt.imshow(img[i], cmap='gray')
#    plt.pause(1)
# plt.clf()
# plt.close()



class SelfLearning(Environment):
   def __init__(self):
      super().__init__()
      self.inference = Inference()
      self.grasp_decision = GraspDecision()

      self.episode = 0
      self.image_states = ["grasp", "place_b", "goal", "place_a"]
      self.percentage_secondary = 0.1
      self.primary_selection_method = SelectionMethod.Max
      self.secondary_selection_method = SelectionMethod.Prob

   def manipulate(self):
      data = {}
      while self.episode < 50000:
         print(self.episode)
         if self.episode < 2000:
            method = SelectionMethod.Random
         else:
            method = self.secondary_selection_method if np.random.rand() > self.percentage_secondary else SelectionMethod.Random

            # failure
            # method = SelectionMethod.Max if np.random.rand() > self.percentage_secondary else SelectionMethod.Top5


            
         # TODO: get camera images

         # grasp images
         grasp_imgs = self.plot_env(episode = self.episode, num_obj = 3, image_state = self.image_states[0])

         # goal images
         dir = "./data/obj_info/obj_info" + str(self.episode) + ".json"
         with open(dir, mode="rt", encoding="utf-8") as f:
            obj_infos = json.load(f)
         goal_imgs = self.plot_env(episode = self.episode, num_obj = 1, image_state = self.image_states[2], obj_info = obj_infos["0"])
         
         # place_b images
         place_b_imgs = self.plot_env(episode = self.episode, num_obj = 0, image_state = self.image_states[1])
         actions = self.inference.infer(grasp_imgs[1], goal_imgs[1], method, place_images=place_b_imgs[1])
         # TODO: planning grasp_trajectry
         reward = 0
         # decicsion success
         for obj_info in obj_infos:

            grasp_execute = self.grasp_decision.is_cheked_grasping(actions["grasp"], obj_infos[str(obj_info)])
            if grasp_execute:
               place_obj_info = obj_infos[str(obj_info)]
               reward = 1
               break
         actions["grasp"]["reward"] = reward

         if grasp_execute:
            # TODO:planning place_trajectry
            place_a_imgs = self.plot_env(episode = self.episode, num_obj = 1, image_state = self.image_states[3], action=actions["place"]["pose"], obj_info = place_obj_info)
            reward = 1
            print("place_success")
         else:
            place_a_imgs = self.plot_env(episode = self.episode, num_obj = 0, image_state = self.image_states[3])
            reward = 0
         actions["place"]["reward"] = reward
         

         # save data
         data[str(self.episode)] = actions
         path = './data/datasets/datasets' + '' + '.json'
         json_file = open(path, mode="w")
         json.dump(data, json_file, ensure_ascii=False)
         json_file.close()

         # learning
         self.retrain_model()

         self.episode += 1


   def retrain_model(self) -> None:
      train_script = Path.home() / '2D-sim' / 'scripts' / 'learning' / 'train.py'
      process = Popen([sys.executable, str(train_script)])
      process.communicate()


learn = SelfLearning()
learn.manipulate()