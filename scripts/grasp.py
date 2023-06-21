#!/usr/bin/env python3
import os
import sys
from subprocess import Popen
import time

import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
from pathlib import Path

from learning.train import Train
from inference.grasp_inference import Inference
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
      self.load_model_args =  [] # train
      self.episode = 0
      self.image_states = ["grasp", "place_b", "goal", "place_a"]
      self.percentage_secondary = 0.1
      self.primary_selection_method = SelectionMethod.Max
      self.secondary_selection_method = SelectionMethod.Prob
      self.previous_model_timestanp=""

   def manipulate(self):
      data = {}

      while self.episode < 100000:
         print(self.episode)
         if self.episode < 2000:
            method = SelectionMethod.Random
         else:
            method = self.secondary_selection_method if np.random.rand() > self.percentage_secondary else SelectionMethod.Random

            # failure
            # method = SelectionMethod.Max if np.random.rand() > self.percentage_secondary else SelectionMethod.Top5


            
         # TODO: get camera images

         # grasp images
         grasp_imgs = self.plot_env(episode = self.episode, num_obj = 1, image_state = self.image_states[0])
         dir = "./data/obj_info/obj_info" + str(self.episode) + ".json"
         with open(dir, mode="rt", encoding="utf-8") as f:
            obj_infos = json.load(f)
         reward = 0
         # decicsion success
         actions = self.inference.infer(grasp_imgs[1], method = method, episode=self.episode)
         for obj_info in obj_infos:
            grasp_execute = self.grasp_decision.is_cheked_grasping(actions["grasp"], obj_infos[str(obj_info)])
            if grasp_execute:
               place_obj_info = obj_infos[str(obj_info)]
               reward = 1
               break
         actions["grasp"]["reward"] = reward
         
         # save data
         data[str(self.episode)] = actions
         path = './data/datasets/datasets' + '_grasp' + '.json'
         json_file = open(path, mode="w")
         json.dump(data, json_file, ensure_ascii=False)
         json_file.close()

         # learning
         if self.episode == 10:
            self.load_model_args = ['--load_model']
         self.retrain_model()

         self.episode += 1
         time.sleep(1)

   def retrain_model(self) -> None:
      cmd = '/root/2D-sim/scripts/learning/grasp_train.py'
      process = Popen(["python3", cmd] + self.load_model_args)
      process.communicate()



learn = SelfLearning()
learn.manipulate()