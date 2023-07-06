import cv2
from loguru import logger
import numpy as np
import copy
import random
import pickle

import torch
from torch.utils.data import Dataset

from utils.image import  get_area_of_interest_new


class CustomDataset():
   def __init__(self, episodes, seed=None):
      super().__init__()
      self.keys = list(episodes.keys())
      self.episodes = episodes


      self.size_input = (480, 752)
      self.size_memory_scale = 4
      self.size_cropped = (300, 300)
      self.size_result = (32, 32)

      self.size_cropped_area = (self.size_cropped[0] // self.size_memory_scale, self.size_cropped[1] // self.size_memory_scale)



      self.box_distance = 0.281  # [m]

   #   self.indexer = GraspIndexer([0.05, 0.07, 0.086])  # [m]
      self.indexer = ([0.025, 0.05, 0.07, 0.086])  # [m]

      self.img_type = "depth"


      self.seed = seed
      self.random_gen = np.random.RandomState(seed)

   def load_image(self, episode_id, action_id):
      image = cv2.imread("./data/img/" + self.img_type + "_" + action_id + str(episode_id) + ".png", cv2.IMREAD_UNCHANGED)
      image = cv2.resize(image, (self.size_input[0] // self.size_memory_scale, self.size_input[1] // self.size_memory_scale))
      return image

   def area_of_interest(self, image, pose):
      area = get_area_of_interest_new(
         image,
         pose,
         size_cropped=self.size_cropped_area,
         size_result=self.size_result,
         size_memory_scale = self.size_memory_scale,
      )
      if len(area.shape) == 2:
         return np.expand_dims(area, 2)
      return area

   def generator(self, index):
      e = self.episodes[index]

      result = []

      grasp = e['grasp']
      # TODO self.img_type 
      grasp_before = self.load_image(index, 'grasp')
      grasp_before_area = self.area_of_interest(grasp_before, grasp['pose'])
      reward_grasp = grasp["reward"]
      # TODO: grasp width
      grasp_index = grasp['index']

      return np.array(grasp_before_area.transpose(2, 0, 1), dtype=np.float32), np.array(reward_grasp, dtype=np.float32)
      

      

   def torch_generator(self, index):
      r = self.generator(index)
      r = tuple(torch.from_numpy(arr) for arr in r)
      return r

   def get_data(self, data):
      # while len(data) > 20000:
      #    data.pop(0)

      if len(data) == 0:
         for key in self.keys:
            r = self.torch_generator(key)
            data.append(((r[0]), (r[1])))

      else:
         r = self.torch_generator(self.keys[-1])
         data.append(((r[0]), (r[1])))

      return data
