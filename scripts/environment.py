#!/usr/bin/env python3
import time
import pathlib

import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
import json

class Environment:
   def __init__(self, size = (752, 480)):
      self.rgb_size = (size[0], size[1], 3)
      self.depth_size = (size[0], size[1], 1)

      self.obj_forms = ["rectangle", "circle"]
      self.size_range = [75, 100] # [min,max]
      self.rec_size = [50, 80]
      self.cir_size = [40]

      # np.random.seed(seed=1)

   def plot_env(self, episode:int = 0, num_obj: int = 1, image_state:str = "grasp", display:bool=False, action:np.array=None, obj_info:dict=None, path:str="./data"):
      path_dir = pathlib.Path(path)

      rgb_img = np.ones(self.rgb_size) * 255
      rgb_img = np.clip(rgb_img, 0, 255).astype(np.uint8)
      depth_img = np.zeros(self.depth_size)

      infos = {}
      for idx in range(num_obj):
         rgb_img, depth_img, info = self.draw_object(rgb_img, depth_img, action = action, obj_info=obj_info)
         infos[idx] = info
      
      plt.show(block=False)
      plt.gca().axis("off")
      if display:
         # plt.imshow(rgb_img)
         # plt.pause(0.5)
         plt.imshow(depth_img, cmap='gray')
         plt.pause(0.5)
         plt.clf()
         plt.close()
      else:
         # plt.imshow(rgb_img)
         plt.imshow(depth_img, cmap='gray')

      # cv2.imwrite(str(path_dir.joinpath('img/rgb_' + image_state + str(episode) + '.png')), rgb_img)
      cv2.imwrite(str(path_dir.joinpath('img/depth_' + image_state + str(episode) + '.png')), depth_img)

      #  
      if image_state == "grasp":
         # path = path_dir.joinpath('obj_info/obj_info' + str(episode) + '.json')
         path = path_dir.joinpath('obj_info/obj_info.json')
         json_file = open(path, mode="w")
         json.dump(infos, json_file, ensure_ascii=False)
         json_file.close()

      return [rgb_img.astype(np.uint8), depth_img.astype(np.uint8)]


   def draw_object(self, rgb_img:np.array = None, depth_img:np.array = None, action:np.array = None, obj_info:list = None):
      if isinstance(obj_info,type(None)):
         # obj_form = self.obj_forms[np.random.randint(0, 2, 1)[0]]
         obj_form = self.obj_forms[0]
         color = np.random.randint(0, 255, 3).tolist()
         depth = np.random.randint(40, 45, 1).tolist()
      else:
         obj_form = obj_info["form"]
         color = obj_info["color"]
         depth = obj_info["depth"]


      output = self.object_pose(obj_form, action, obj_info)


      if obj_form == "rectangle":
         obj_cpose, angle, points, obj_size = output[0], output[1], output[2], output[3]
         cv2.fillConvexPoly(rgb_img, points, color)
         cv2.fillConvexPoly(depth_img, points, depth)

      elif obj_form == "circle":
         points = None
         angle = None
         obj_cpose, obj_size = output[0], output[1]
         cv2.circle(rgb_img, obj_cpose, obj_size[0], color, -1)
         cv2.circle(depth_img, obj_cpose, obj_size[0], depth, -1)

      else:
         print("Unanticipated input: ", obj_form)
      
      if isinstance(action, type(None)):
         info = {"form":obj_form,
               "color":color,
               "depth":depth,
               "size":obj_size.tolist(),
               "center_psoe":obj_cpose.tolist(),
               "angle":angle,
               "points":points if isinstance(points,type(None)) else points.tolist()}
      else:
         info = []

      return rgb_img, depth_img, info


   def object_pose(self, obj_form, action=None, obj_info=None):
      pose_range = np.array([self.rgb_size[1], self.rgb_size[0]]) - np.array([self.size_range[1], self.size_range[1]])

      if obj_form == "rectangle":
         # obj_size = np.random.randint(self.size_range[0], self.size_range[1], 2) if isinstance(obj_info, type(None)) else np.array(obj_info["size"])
         obj_size = np.array(self.rec_size) if isinstance(obj_info, type(None)) else np.array(obj_info["size"])

         if isinstance(action, type(None)):
            obj_cpose = np.multiply(np.random.rand(2), pose_range)
            obj_cpose += np.array([self.size_range[1]/2, self.size_range[1]/2])
            angle = np.random.rand() * math.pi - math.pi/2
         else:
            obj_cpose = np.array(action[:2])
            angle = action[2]

         left = - obj_size[0] / 2
         right = + obj_size[0] / 2
         top = - obj_size[1] / 2
         bottom = + obj_size[1] / 2

         R = np.array([[np.cos(-angle), np.sin(-angle)],
                    [-np.sin(-angle), np.cos(-angle)]])

         points = np.append([R.dot([top,left])], [R.dot([top,right])], axis = 0)
         points = np.append(points, [R.dot([bottom,right])],  axis = 0)
         points = np.append(points, [R.dot([bottom,left])], axis = 0)
         points += obj_cpose

         return obj_cpose.astype(np.int64), angle, points.astype(np.int64), obj_size

      elif obj_form == "circle":
         # obj_size = np.random.randint(self.size_range[0]/2, self.size_range[1]/2, 1) if isinstance(obj_info, type(None)) else np.array(obj_info["size"])
         obj_size = np.array(self.cir_size) if isinstance(obj_info, type(None)) else np.array(obj_info["size"])
         if isinstance(action, type(None)):
            obj_cpose = np.multiply(np.random.rand(2), pose_range)
            obj_cpose += np.array([self.size_range[1]/2, self.size_range[1]/2])
         else:
            obj_cpose = np.array(action[:2])

         return obj_cpose.astype(np.int64), obj_size

      else:
         print("Unanticipated input: ", obj_form)
        


if __name__ == "__main__":
   episode = 0

   env = Environment(size = (752, 480))
   # output=env.plot_env(episode = episode, num_obj = 3, image_state = "grasp")
   # print(output[0].shape)