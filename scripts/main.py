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
from inference.inference import Inference
from inference.inference_utils import InferenceUtils
from environment import Environment
from utils.param import Mode, SelectionMethod
from action.grasp_decision import GraspDecision
from learning.train import Train


class SelfLearning(Environment):
   def __init__(self):
      super().__init__()
      self.inference = Inference()
      self.grasp_decision = GraspDecision()
      self.load_model_args =  False
      self.episode = 0
      self.image_states = ["grasp", "place_b", "goal", "place_a"]
      self.percentage_secondary = 0.0
      self.primary_selection_method = SelectionMethod.Max
      self.secondary_selection_method = SelectionMethod.Prob
      self.previous_model_timestanp=""

   def manipulate(self):
      data = {}
      path = './data/datasets/datasets' + '' + '.json'
      if self.episode > 0:
         with open(path, mode="rt", encoding="utf-8") as f:
            data = json.load(f)
      while self.episode < 100000:
         print(self.episode)
         if self.episode < 50:
            method = SelectionMethod.Random
         else:
            method = self.primary_selection_method if np.random.rand() > self.percentage_secondary else self.secondary_selection_method

            # failure
            # method = SelectionMethod.Max if np.random.rand() > self.percentage_secondary else SelectionMethod.Top5

            
         # TODO: get camera images

         # grasp images
         grasp_imgs = self.plot_env(episode = self.episode, num_obj = 1, image_state = self.image_states[0])

         # goal images
         # dir = "./data/obj_info/obj_info" + str(self.episode) + ".json"
         dir = "./data/obj_info/obj_info.json"
         with open(dir, mode="rt", encoding="utf-8") as f:
            obj_infos = json.load(f)
         img_num = np.random.randint(1, 10, 1)
         img_name = "rec_goal" if obj_infos["0"]["form"] == "rectangle" else "cir_goal"
         goal_img = cv2.imread("./data/goal/" + img_name + str(img_num[0]) + ".png", cv2.IMREAD_UNCHANGED)

         # place_b images
         place_b_imgs = self.plot_env(episode = self.episode, num_obj = 0, image_state = self.image_states[1])
         actions = self.inference.infer(grasp_imgs[1], goal_img, method, place_images=place_b_imgs[1], episode=self.episode)
         # TODO: planning grasp_trajectry

         reward = 0
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
         json_file = open(path, mode="w")
         json.dump(data, json_file, ensure_ascii=False)
         json_file.close()

         # learning
         if self.episode > 10:
            self.load_model_args = True

         if self.episode > 5:
            train = Train(
               image_format="png",
               data_path="/root/2D-sim/scripts/data",
               load_model=self.load_model_args
            )

         self.episode += 1



learn = SelfLearning()
learn.manipulate()