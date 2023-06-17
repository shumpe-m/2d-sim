import cv2
from loguru import logger
import numpy as np
import copy
import random

import torch
from torch.utils.data import Dataset

from utils.image import  get_area_of_interest_new


class CustomDataset(Dataset):
   def __init__(self, episodes, seed=None):
      super().__init__()
      self.episodes = episodes
      self.episodes_place_success_index = []
      for i in range(len(episodes)):
         if episodes[str(i)]['place']['reward'] > 0:
            self.episodes_place_success_index.append(i)



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
      e = self.episodes[str(index)]
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

      place = e['place']
      place_before = self.load_image(index, 'place_b')
      place_after = self.load_image(index, 'place_a')

      # Generate goal has no action_id
      def generate_goal(g_suffix, g_pose, g_suffix_before='v', g_reward=0, g_index=None, g_place_weight=1.0, g_merge_weight=1.0, jitter=None):
         if g_suffix == 'v' and g_suffix_before == 'v':
               place_goal_before = place_before
               place_goal = place_before
         elif g_suffix == 'v' and g_suffix_before == 'after':
               place_goal_before = place_after
               place_goal = place_before
         elif g_suffix == 'after' and g_suffix_before == 'v':
               place_goal_before = place_before
               place_goal = place_after
         elif g_suffix == 'after' and g_suffix_before == 'after':
               place_goal_before = place_after
               place_goal = place_after
         else:
               goal_e = self.episodes[g_index]

               g_pose = g_pose if g_pose else goal_e['place']['pose']

               place_goal_before = self.load_image(self.img_type, index, 'grasp')
               place_goal = self.load_image(g_collection, g_episode_id, 1, 'ed-' + g_suffix)

         if isinstance(jitter, dict):
               g_pose = self.jitter_pose(g_pose, **jitter)

         place_before_area = self.area_of_interest(place_goal_before, g_pose)
         place_goal_area = self.area_of_interest(place_goal, g_pose)

         reward_grasp = grasp['reward']
         reward_place = g_reward * grasp['reward'] * place['reward']
         reward_merge = reward_place

         grasp_weight = g_reward
         place_weight = (1.0 + 3.0 * reward_place) * reward_grasp * g_place_weight
         merge_weight = (1.0 + 3.0 * reward_merge) * reward_grasp * g_merge_weight

         return (
               grasp_before_area.transpose(2, 0, 1),
               place_before_area.transpose(2, 0, 1),
               place_goal_area.transpose(2, 0, 1),
               (reward_grasp, reward_place, reward_merge),
               (grasp_index, 0, 0),
               (grasp_weight, place_weight, merge_weight),
         )

      if self.use_hindsight:
         result.append(generate_goal('after', place['pose'], g_reward=1))

         result += [
               generate_goal('after', place['pose'], jitter={})
               for _ in range(self.jittered_hindsight_images)
         ]

         if place['reward'] > 0:
               result += [
                  generate_goal('after', place['pose'], jitter={'scale_x': 0.02, 'scale_y': 0.01, 'scale_a': 0.2})
                  for _ in range(self.jittered_hindsight_x_images)
               ]
      #TODO some actions
      if self.use_further_hindsight and 'bin_episode' in place:
         for i in range(index + 1, len(self.episodes)):
               place_later = self.episodes[i]['actions'][1]
               if place_later['bin_episode'] != place['bin_episode']:
                  break

               if place_later['reward'] > 0:
                  result.append(generate_goal(None, None, 'after', place['pose'], g_index=i, g_reward=1))

      if self.use_negative_foresight:
         g_suffix, g_suffix_before = random.choice([('v', 'v'), ('after', 'after'), ('v', 'after')])
         result.append(generate_goal(g_suffix, place['pose'], g_suffix_before=g_suffix_before, jitter={'around': False}))
      # TODO
      if self.use_own_goal and 'ed-goal' in place:
         result.append(generate_goal('goal', place['pose'], g_place_weight=0.2, g_merge_weight=0.7, g_index=index))

         result += [
               generate_goal('goal', place['pose'], g_index=index, jitter={})
               for _ in range(self.jittered_goal_images)
         ]

      if self.use_different_episodes_as_goals and self.episodes_place_success_index!=[]:
         result += [
               generate_goal('after', None, g_index=goal_index, g_place_weight=0.0)
               for goal_index in self.random_gen.choice(self.episodes_place_success_index, size=self.different_episodes_images)
         ]

         if place['reward'] > 0:
               result += [
                  generate_goal('after', None, g_index=goal_index, g_place_weight=0.0)
                  for goal_index in self.random_gen.choice(self.episodes_place_success_index, size=self.different_episodes_images_success)
               ]

               # for k, v in self.episodes_different_objects_ids.items():
               #    if v[0] <= e['id'] <= v[1]:
               #       result += [
               #             generate_goal(None, None, 'after', None, g_index=goal_index, g_place_weight=0.0)
               #             for goal_index in self.random_gen.choice(self.episodes_different_objects_index[k], size=self.different_object_images)
               #       ]

               #       # result += [
               #       #     generate_goal(None, None, 'after', None, g_index=goal_index, jitter={})
               #       #     for goal_index in self.random_gen.choice(self.episodes_different_objects_index[k], size=self.different_jittered_object_images)
               #       # ]

      return [np.array(t, dtype=np.float32) for t in zip(*result)]

   def torch_generator(self, index):
      r = self.generator(index)
      r = tuple(torch.from_numpy(arr) for arr in r)
      return (r[0], r[1], r[2]), (r[3], r[4], r[5])

   def __len__(self):
      return len(self.episodes)

   def __getitem__(self, index):
      return self.torch_generator(index)