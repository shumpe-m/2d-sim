import cv2
from loguru import logger
import numpy as np
import copy
import random

import torch
from torch.utils.data import Dataset

from utils.image import  get_area_of_interest_new


class CustomDataset():
   def __init__(self, episodes, seed=None):
      super().__init__()
      self.keys = list(episodes.keys())
      self.keys = self.keys[-13000:]
      self.episodes_place_success_index = []
      self.episodes = {}
      for key in self.keys:
         self.episodes[key] = episodes[key]



      self.size_input = (480, 752)
      self.size_memory_scale = 4
      self.size_cropped = (200, 200)
      self.size_result = (32, 32)

      self.size_cropped_area = (self.size_cropped[0] // self.size_memory_scale, self.size_cropped[1] // self.size_memory_scale)

      self.use_hindsight = True
      self.use_further_hindsight = False
      self.use_negative_foresight = True
      self.use_own_goal = True
      self.use_different_episodes_as_goals = True

      self.jittered_hindsight_images = 1
      self.jittered_hindsight_x_images = 2  # Only if place reward > 0
      self.jittered_goal_images = 1
      self.different_episodes_images = 1
      self.different_episodes_images_success = 4  # Only if place reward > 0
      self.different_object_images = 4  # Only if place reward > 0
      self.different_jittered_object_images = 0  # Only if place reward > 0

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

   def jitter_pose(self, pose, scale_x=0.05, scale_y=0.05, scale_a=1.5, around=True):
      new_pose = copy.deepcopy(pose)

      if around:
         low = [np.minimum(0.001, scale_x), np.minimum(0.001, scale_y), np.minimum(0.06, scale_a)]
         mode = [np.minimum(0.006, scale_x), np.minimum(0.006, scale_y), np.minimum(0.32, scale_a)]
         high = [scale_x + 1e-6, scale_y + 1e-6, scale_a + 1e-6]
         dx, dy, da = self.random_gen.choice([-1, 1], size=3) * self.random_gen.triangular(low, mode, high, size=3)
      else:
         low = [-scale_x - 1e-6, -scale_y - 1e-6, -scale_a - 1e-6]
         mode = [0.0, 0.0, 0.0]
         high = [scale_x + 1e-6, scale_y + 1e-6, scale_a + 1e-6]
         dx, dy, da = self.random_gen.triangular(low, mode, high, size=3)

      new_pose[0] += np.cos(pose[2]) * dx - np.sin(pose[2]) * dy
      new_pose[1] += np.sin(pose[2]) * dx + np.cos(pose[2]) * dy
      new_pose[2] += da
      return new_pose

   def generator(self, index):
      e = self.episodes[index]

      result = []

      grasp = e['grasp']
      # TODO self.img_type 
      grasp_before = self.load_image(index, 'grasp')
      grasp_before_area = self.area_of_interest(grasp_before, grasp['pose'])
      # TODO: grasp width
      grasp_index = 1

      # Only single grasp
      if len(e) == 1:
         pass


      # Generate goal has no action_id
      def generate_goal():


         reward_grasp = grasp['reward']


         grasp_weight = grasp['reward']

         return (
               grasp_before_area.transpose(2, 0, 1),
               (reward_grasp),
               (grasp_index),
               (grasp_weight),
         )

      if self.use_hindsight:
         result.append(generate_goal())

      return [np.array(t, dtype=np.float32) for t in zip(*result)]

   def torch_generator(self, index):
      r = self.generator(index)
      r = tuple(torch.from_numpy(arr) for arr in r)
      return r

   def get_data(self):
      
      data = []
      for key in self.keys:
         r = self.torch_generator(key)
         for b_dx in range(r[0].shape[0]):
            data.append(((r[0][b_dx]), (r[1][b_dx], r[2][b_dx], r[3][b_dx])))


      return data
