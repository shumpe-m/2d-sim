import time
from typing import List

import cv2
# from loguru import logger
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from itertools import product
import json

from inference.inference_utils import InferenceUtils
from utils.param import SelectionMethod
# from models.models import GraspModel, PlaceModel, MergeModel
from models.models_sub import GraspModel


class Inference(InferenceUtils):
   def __init__(self, lower_random_pose=[0.5, 0.5, -1.484], upper_random_pose=[480.5, 752.5, 1.484]):
      super(Inference, self).__init__(
         lower_random_pose=lower_random_pose,
         upper_random_pose=upper_random_pose
      )
      self.number_top_grasp = 200
      self.number_top_place = 200
      self.input_shape = [None, None, 1] if True else [None, None, 3]
      self.z_shape = 48
      # TODO:load model
      self.grasp_model = GraspModel(self.input_shape[2]).float()

      self.previous_model_timestanp = ""
      self.reload_model_weights()

   def reload_model_weights(self, lode_model = True):
      with open('./data/checkpoints/grasp_timestamp.txt', 'r') as f:
         saved_timestamp = f.read()
      if lode_model and self.previous_model_timestanp!=saved_timestamp:
         cptfile = './data/checkpoints/grasp_out.cpt'
         cpt = torch.load(cptfile)
         self.grasp_model.load_state_dict(cpt['grasp_model_state_dict'])


      self.previous_model_timestanp = saved_timestamp

   def infer(
         self,
         images,
         goal_images = None,
         method = None,
         verbose=1,
         place_images = None,
         episode = None,
      ):
      self.reload_model_weights()
      start = time.time()
      actions = {}
      grasp_action = {}
      place_action = {}
      if method == SelectionMethod.Random:
         dir = "./data/obj_info/obj_info" + str(episode) + ".json"
         with open(dir, mode="rt", encoding="utf-8") as f:
                     obj_infos = json.load(f)
         pose = obj_infos["0"]["center_psoe"]
         angle = obj_infos["0"]["angle"] if obj_infos["0"]["angle"] != None else np.random.uniform(self.lower_random_pose[2], self.upper_random_pose[2])
         grasp_action["index"] = int(np.random.choice(range(3)))

         grasp_action["pose"] = [np.random.uniform(self.lower_random_pose[0], self.upper_random_pose[0]),  # [m]
                                 np.random.uniform(self.lower_random_pose[1], self.upper_random_pose[1]),  # [m]
                                 np.random.uniform(self.lower_random_pose[2], self.upper_random_pose[2])]  # [rad]
         grasp_action["estimated_reward"] = -1
         grasp_action["step"] = 0
         actions["grasp"] = grasp_action
         return actions

      # input_images = [self.get_images(i) for i in images]
      # goal_input_images = [self.get_images(i) for i in goal_images]
      # place_input_images = [self.get_images(i) for i in place_images]
      input_images = self.get_images(images)


      # print(np.array(grasp_input).shape)
      input_images = torch.tensor(input_images)
      input_images = torch.permute(input_images, (0, 3, 1, 2)).float()


      self.grasp_model.eval()


      z_g, reward_g = self.grasp_model(input_images)
      z_g = torch.permute(z_g, (0, 2, 3, 1)).float()
      reward_g = torch.permute(reward_g, (0, 2, 3, 1)).float()

      first_method = SelectionMethod.PowerProb if method in [SelectionMethod.Top5, SelectionMethod.Max] else method

      filter_lambda_n_grasp = self.get_filter_n(first_method, self.number_top_grasp)

      np_reward_g = reward_g.to('cpu').detach().numpy().copy()

      g_top_index = filter_lambda_n_grasp(np_reward_g)


      g_top_index_unraveled = np.transpose(np.asarray(np.unravel_index(g_top_index, reward_g.shape)))



      g_top_z = z_g[g_top_index_unraveled[:, 0], g_top_index_unraveled[:, 1], g_top_index_unraveled[:, 2]]





      reward = rewards.to('cpu').detach().numpy().copy()

      

      filter_measure = reward_g


      filter_lambda = self.get_filter(method)
      index_raveled = filter_lambda(filter_measure)
      # print(index_raveled)

      index_unraveled = np.unravel_index(index_raveled, reward_g.shape)
      g_index = g_top_index_unraveled[index_unraveled[0]]



      grasp_action["index"] = int(g_index[3])
      grasp_action["pose"] = self.pose_from_index(g_index, reward_g.shape, images[0])
      grasp_action["estimated_reward"] = int(reward_g[tuple(g_index)])
      grasp_action["step"] = 0


   
      # if verbose:
      #    logger.info(f'NN inference time [s]: {time.time() - start:.3}')
      actions["grasp"] = grasp_action
      return actions
      

class CombinedDataset(Dataset):
    def __init__(self, data1, data2):
        self.data1 = data1
        self.data2 = data2
        self.combinations = list(product(range(len(data1)), range(len(data2))))

    def __getitem__(self, index):
        index1, index2 = self.combinations[index]
        return self.data1[index1], self.data2[index2], index1, index2

    def __len__(self):
        return len(self.combinations)