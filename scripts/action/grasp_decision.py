import pathlib

import numpy as np
import math

class GraspDecision:
   def __init__(self):
      self.gripper = [40, 20, 10]
      self.ang_range = [np.sin(math.pi/2), np.sin(math.pi/6), np.sin(math.pi/10)]
      self.radius = 40

   def is_cheked_grasping(self, action, obj_info):
      pose = [None, None, None]
      center = np.array(obj_info["center_psoe"])
      index = action["index"]

      if isinstance(obj_info["angle"], type(None)):
         a_succesee = True
         pose = [(action["pose"][0] - center[0]) * math.cos(-action["pose"][2]) + (action["pose"][1] - center[1]) * math.sin(-action["pose"][2]) * -1, 
            (action["pose"][0] - center[0]) * math.sin(-action["pose"][2]) + (action["pose"][1] - center[1]) * math.cos(-action["pose"][2]), 
            action["pose"][2]]

         in_area = True if abs(pose[1]) <= self.gripper[index] and abs(pose[0]) <= 5 else False
         distance = 0 if in_area else 50
         
      else:
         obj_angle = obj_info["angle"]
         pose = [(action["pose"][0] - center[0]) * math.cos(-obj_angle) + (action["pose"][1] - center[1]) * math.sin(-obj_angle) * -1, 
            (action["pose"][0] - center[0]) * math.sin(-obj_angle) + (action["pose"][1] - center[1]) * math.cos(-obj_angle), 
            action["pose"][2]]

         angle_diff = pose[2] + obj_angle
         in_area = True if abs(pose[0]) <= self.gripper[index] - 37 * abs(np.sin(angle_diff)) and \
            abs(pose[1]) <= (40 / self.gripper[0]) * self.gripper[index] - (self.gripper[index]*0.8) * abs(np.cos(angle_diff)) \
            else False

         a_succesee = False if np.sin(angle_diff) > self.ang_range[index] or \
            (math.pi/4.25 <= pose[2] + obj_angle and math.pi/3.7 >= pose[2] + obj_angle) or \
            (3*math.pi/4.25 <= pose[2] + obj_angle and 3*math.pi/3.7 >= pose[2] + obj_angle) or \
            (-math.pi/4.25 >= pose[2] + obj_angle and -math.pi/3.7 <= pose[2] + obj_angle) or \
            (-3*math.pi/4.25 >= pose[2] + obj_angle and -3*math.pi/3.7 <= pose[2] + obj_angle) else True

         if (math.pi/4.25 <= pose[2] + obj_angle and math.pi/3.7 >= pose[2] + obj_angle) or \
            (3*math.pi/4.25 <= pose[2] + obj_angle and 3*math.pi/3.7 >= pose[2] + obj_angle) or \
            (-math.pi/4.25 >= pose[2] + obj_angle and -math.pi/3.7 <= pose[2] + obj_angle) or \
            (-3*math.pi/4.25 >= pose[2] + obj_angle and -3*math.pi/3.7 <= pose[2] + obj_angle):
            print("###############################")
            print(pose[2] * 180 / 3.1415, obj_angle * 180 / 3.1415)
            print((pose[2] + obj_angle) * 180 / 3.1415)
            # print(abs(angle_diff) >= math.pi/4.7 and abs(angle_diff) <= math.pi/3.3)
            if in_area:
               print("$$$$$$$$$$$$$")



      index_success = False if index == 3 else True 
      # print(distance)
      # a_succesee = True

      execute = True if in_area and a_succesee and index_success else False
      # print("action:"+str(action)+"  target:"+str(obj_info["center_psoe"])+str(obj_info["angle"]) + "  distance:"+str(distance)+"  :"+str(angle_diff))
      return execute

      