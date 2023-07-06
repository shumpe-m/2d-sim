import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from inference.inference import Inference
from models.models import GraspModel, PlaceModel, MergeModel


class Heatmap:
   def __init__(self, model, a_space=None):
      self.model = model
      self.model.eval()
      self.inf = Inference()
      if a_space is not None:
         self.inf.a_space = a_space
      self.device = "cuda" if torch.cuda.is_available() else "cpu"

   def calculate_heat(self, reward):
      size_input = (480, 752)
      size_reward_center = (reward.shape[1] / 2, reward.shape[2] / 2)
      scale = 300 / 32 * (80.0 / reward.shape[1])

      a_space_idx = range(len(self.inf.a_space))

      heat_values = np.zeros(size_input[::-1], dtype=float)
      for i in a_space_idx:
         a = self.inf.a_space[i]
         rot_mat = cv2.getRotationMatrix2D(size_reward_center, -a * 180.0 / np.pi, scale)
         rot_mat[0][2] += size_input[0] / 2 - size_reward_center[0]
         rot_mat[1][2] += size_input[1] / 2 - size_reward_center[1]
         heat_values += cv2.warpAffine(reward[i], rot_mat, size_input, borderValue=0)

      norm = (5 * heat_values.max() + len(a_space_idx)) / 6
      # norm = heat_values.max()

      return heat_values * 255.0 / norm

   def render(
         self,
         image,
         goal_image = None,
         alpha=0.2,
         save_path=None,
         reward_index=None,
         draw_directions=False,
         indices=None,
      ):
      base = image
      image = self.inf.get_images(image)
      # for i in range(image.shape[0]): 
      #    plt.show(block=False)
      #    plt.gca().axis("off")
      #    plt.imshow(image[i], cmap='gray')
      #    plt.pause(0.3)
      #    plt.clf()
      #    plt.close()
      input_images = torch.tensor(image)
      input_images = torch.permute(input_images, (0, 3, 1, 2)).float().to(self.device)

      if isinstance(goal_image, type(None)):
         _, reward = self.model(input_images)
         # print(torch.max(reward))
         # print(torch.min(reward))
      else:
         base = goal_image
         image_goal = self.inf.get_images(goal_image)
         goal_input_images = torch.tensor(image_goal)
         goal_input_images = torch.permute(goal_input_images, (0, 3, 1, 2)).float().to(self.device)
         _, reward = self.model(input_images, goal_input_images)
         
      if reward_index is not None:
         _. reward = reward[reward_index]

      # reward = np.maximum(reward, 0)
      reward = reward.to('cpu').detach().numpy().copy()
      reward_mean = np.mean(reward, axis=1)
      # reward_mean = reward[:, :, :, 0]

      heat_values = self.calculate_heat(reward_mean)

      heatmap = cv2.applyColorMap(heat_values.astype(np.uint8), cv2.COLORMAP_JET)

      base_heatmap = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB) / 255 + (1 - alpha) * heatmap
      base = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)
      base_heatmap = cv2.addWeighted(base.astype(float), 1 - alpha, base_heatmap.astype(float), alpha, 0)

      if save_path:
         cv2.imwrite(str(save_path), base_heatmap)
      return base_heatmap


if __name__ == '__main__':
   model_type = "grasp"

   if model_type == "merge":
      save_path = "./data/heatmaps"
      device = "cuda" if torch.cuda.is_available() else "cpu"

      image_grasp = cv2.imread("./data/img/depth_grasp3.png", cv2.IMREAD_GRAYSCALE)
      image_place = cv2.imread("./data/img/depth_place_b3.png", cv2.IMREAD_GRAYSCALE)
      image_goal = cv2.imread("./data/goal/rec_goal1.png", cv2.IMREAD_GRAYSCALE)


      grasp_model = GraspModel(1).float().to(device)
      place_model = PlaceModel(1*2).float().to(device)
      cptfile = './data/checkpoints/grasp_model.cpt'
      cpt = torch.load(cptfile)
      grasp_model.load_state_dict(cpt['grasp_model_state_dict'])
      place_model.load_state_dict(cpt['place_model_state_dict'])

      indices = np.array([[22, 25, 18, 1], [13, 21, 17, 2], [7, 19, 16, 2], [11, 22, 17, 2], [15, 10, 12, 2], [20, 24, 15, 2], [17, 23, 15, 0], [0, 16, 15, 1], [15, 22, 15, 0], [6, 17, 16, 1], [20, 10, 20, 1], [32, 15, 12, 0], [28, 26, 21, 1], [30, 26, 20, 1], [14, 12, 24, 1], [18, 11, 21, 1], [12, 11, 24, 0], [9, 14, 26, 1], [19, 10, 19, 0], [11, 12, 26, 0], [22, 24, 19, 2], [4, 16, 17, 2], [12, 11, 25, 1], [23, 12, 17, 0], [28, 19, 10, 2], [22, 25, 17, 2], [18, 11, 23, 1], [15, 11, 25, 1], [22, 24, 17, 1], [14, 23, 15, 1], [0, 15, 16, 2], [13, 20, 17, 2], [9, 19, 14, 1], [18, 9, 20, 1], [15, 12, 24, 2], [2, 17, 13, 1], [16, 10, 23, 2], [16, 10, 21, 0], [16, 23, 14, 2], [4, 18, 16, 2], [9, 18, 16, 2], [25, 17, 9, 1], [9, 19, 16, 2], [12, 9, 16, 1], [16, 11, 24, 1], [23, 10, 18, 1], [22, 11, 18, 0], [19, 9, 20, 0], [3, 17, 16, 1], [7, 19, 15, 1], [9, 20, 16, 2], [28, 13, 14, 0], [11, 20, 16, 1], [24, 25, 16, 2], [14, 12, 23, 0], [24, 10, 18, 1], [2, 15, 16, 2], [15, 9, 22, 0], [13, 12, 24, 2], [14, 21, 14, 1], [15, 10, 24, 0], [0, 16, 16, 1], [4, 17, 16, 2], [27, 25, 19, 0], [20, 10, 19, 1], [21, 11, 17, 0], [17, 12, 22, 0], [2, 15, 16, 1], [12, 14, 25, 0], [9, 20, 17, 2], [14, 21, 15, 0], [18, 9, 23, 1], [12, 13, 23, 2], [22, 10, 18, 1], [14, 11, 26, 2], [31, 25, 21, 1], [12, 20, 14, 2], [17, 9, 22, 2], [26, 24, 18, 1], [16, 13, 21, 0], [17, 22, 16, 2], [17, 22, 18, 2], [18, 23, 16, 0], [20, 12, 19, 0], [32, 26, 21, 1], [35, 23, 22, 1], [11, 19, 15, 2], [16, 11, 21, 2], [4, 18, 15, 2], [12, 21, 16, 2], [15, 11, 21, 0], [2, 16, 16, 2], [14, 10, 24, 1], [4, 16, 15, 1], [18, 11, 19, 0], [24, 17, 9, 2], [9, 20, 15, 2], [18, 11, 21, 0], [0, 15, 15, 2], [8, 20, 13, 1], [16, 9, 21, 0], [0, 15, 17, 2], [8, 9, 20, 0], [15, 13, 24, 2], [19, 22, 18, 2], [16, 22, 15, 1], [12, 12, 24, 2], [21, 11, 21, 1], [1, 16, 16, 1], [9, 17, 12, 2], [14, 20, 12, 2], [25, 26, 21, 2], [6, 17, 15, 1], [31, 25, 23, 2], [17, 11, 20, 0], [22, 25, 18, 2], [17, 12, 23, 2], [7, 20, 13, 1], [15, 9, 24, 1], [13, 20, 16, 1], [1, 16, 14, 2], [19, 12, 22, 1], [7, 15, 27, 0], [17, 10, 24, 1], [18, 9, 24, 2], [19, 24, 16, 0], [14, 12, 23, 1], [17, 11, 22, 1], [18, 10, 20, 1], [8, 20, 14, 1], [19, 12, 21, 1], [10, 20, 15, 1], [19, 10, 20, 1], [26, 23, 17, 1], [9, 14, 24, 2], [14, 22, 14, 1], [13, 12, 25, 0], [19, 25, 19, 2], [11, 19, 14, 2], [17, 10, 22, 0], [13, 13, 25, 0], [8, 15, 27, 0], [15, 11, 23, 1], [33, 17, 12, 0], [14, 10, 23, 0], [20, 9, 20, 1], [13, 13, 24, 0], [12, 20, 16, 1], [7, 16, 13, 1], [13, 13, 26, 1], [15, 12, 24, 0], [22, 24, 20, 2], [18, 11, 22, 1], [6, 16, 14, 1], [32, 25, 22, 2], [17, 22, 17, 1], [14, 13, 24, 1], [34, 24, 21, 1], [9, 19, 17, 2], [3, 17, 14, 1], [25, 17, 9, 0], [24, 24, 18, 2], [16, 11, 25, 2], [27, 13, 16, 2], [1, 17, 16, 2], [17, 22, 16, 1], [10, 13, 26, 0], [30, 26, 22, 2], [24, 24, 17, 2], [20, 11, 22, 1], [2, 16, 18, 2], [20, 10, 18, 0], [26, 25, 18, 1], [16, 11, 21, 1], [12, 21, 14, 0], [10, 14, 25, 0], [24, 11, 17, 1], [13, 11, 25, 1], [13, 12, 27, 1], [27, 26, 21, 1], [18, 10, 19, 2], [1, 16, 16, 2], [16, 10, 25, 2], [26, 25, 19, 1], [9, 15, 25, 0], [12, 9, 16, 2], [2, 16, 13, 1], [9, 18, 13, 2], [21, 24, 18, 1], [16, 23, 14, 1], [18, 11, 22, 2], [29, 25, 21, 1], [11, 12, 24, 0], [20, 22, 16, 2], [27, 26, 19, 2], [24, 12, 16, 1], [11, 15, 24, 0], [33, 23, 22, 1], [16, 22, 15, 2], [9, 15, 27, 0], [27, 26, 21, 2], [22, 10, 20, 1], [25, 26, 19, 1], [10, 18, 15, 2], [35, 24, 23, 1], [20, 9, 21, 2], [24, 12, 18, 1], [13, 20, 15, 1], [20, 23, 17, 1], [22, 16, 8, 1], [13, 11, 26, 1], [30, 25, 21, 2], [9, 20, 14, 0], [7, 15, 28, 0], [13, 13, 24, 1], [10, 21, 13, 1], [23, 26, 17, 0], [19, 23, 16, 1], [3, 17, 15, 0], [7, 19, 15, 2], [14, 20, 13, 2], [23, 25, 17, 1], [1, 15, 13, 2], [20, 10, 19, 0], [18, 9, 21, 1], [20, 12, 21, 1], [20, 10, 21, 1], [18, 10, 23, 2], [16, 12, 26, 2], [29, 26, 18, 2], [36, 24, 24, 1], [22, 15, 10, 1], [18, 23, 15, 1], [2, 15, 17, 2], [14, 20, 16, 2], [32, 16, 14, 1], [17, 11, 11, 2], [21, 11, 18, 2], [24, 13, 17, 1], [3, 16, 14, 1], [19, 9, 20, 1], [11, 12, 23, 1], [35, 25, 23, 2], [15, 12, 22, 0], [34, 24, 21, 2], [23, 22, 16, 1], [14, 20, 14, 2], [21, 24, 15, 1], [9, 18, 15, 2], [10, 20, 14, 2], [9, 13, 23, 2], [17, 11, 23, 2], [10, 18, 16, 1], [21, 25, 18, 1], [6, 16, 15, 2], [8, 19, 16, 1], [30, 25, 20, 2], [24, 25, 18, 0], [12, 11, 26, 0], [16, 11, 21, 0], [23, 23, 20, 2], [18, 11, 20, 0], [11, 21, 15, 1], [19, 9, 21, 2], [10, 19, 16, 2], [8, 18, 15, 2], [15, 22, 15, 2], [11, 21, 16, 2], [29, 25, 19, 2], [13, 11, 26, 2], [18, 23, 14, 1], [10, 21, 15, 2], [29, 15, 16, 2], [9, 19, 15, 2], [6, 17, 14, 1], [28, 12, 17, 2], [23, 24, 17, 1], [18, 12, 23, 2], [31, 24, 21, 1], [15, 11, 24, 1], [14, 12, 22, 0], [12, 14, 24, 0], [20, 24, 16, 1], [24, 11, 17, 0], [12, 20, 14, 1], [12, 13, 24, 0], [0, 15, 17, 1], [11, 14, 25, 0], [7, 20, 14, 2], [23, 11, 18, 1], [34, 24, 24, 2], [0, 16, 17, 2], [24, 12, 16, 0], [5, 16, 15, 1], [13, 19, 16, 2], [36, 25, 25, 2], [13, 11, 27, 1], [18, 12, 21, 1], [10, 13, 25, 0], [26, 18, 9, 0], [36, 23, 22, 1], [31, 24, 21, 0], [15, 12, 24, 1], [13, 12, 23, 0], [24, 12, 17, 1], [15, 10, 22, 0], [9, 21, 14, 2], [16, 23, 17, 2], [30, 14, 13, 0], [20, 11, 19, 1], [2, 16, 15, 1], [19, 9, 21, 1], [8, 15, 25, 0], [11, 13, 25, 0], [34, 24, 22, 2], [6, 19, 14, 1], [24, 23, 18, 2], [25, 18, 9, 2], [21, 15, 10, 0], [8, 19, 13, 1], [24, 26, 19, 1], [2, 15, 15, 1], [17, 11, 23, 0], [31, 25, 20, 1], [9, 13, 26, 0], [24, 16, 9, 1], [11, 22, 14, 1], [5, 18, 15, 2], [20, 12, 21, 2], [12, 12, 26, 0], [10, 13, 27, 0], [23, 16, 8, 1], [20, 9, 20, 0], [1, 17, 18, 2], [21, 24, 15, 2], [11, 19, 14, 1], [4, 17, 17, 2], [31, 24, 22, 2], [26, 17, 9, 1], [26, 13, 15, 0], [16, 11, 23, 2], [23, 16, 10, 2], [23, 11, 19, 1], [14, 11, 24, 1], [13, 12, 25, 2], [19, 11, 22, 1], [13, 20, 15, 2], [33, 25, 23, 2], [34, 25, 22, 1], [10, 13, 24, 2], [35, 19, 11, 0], [13, 13, 24, 2], [11, 20, 15, 2], [32, 25, 21, 1], [15, 11, 23, 2], [8, 19, 12, 2], [24, 26, 19, 2], [5, 17, 12, 2], [30, 25, 22, 2], [28, 24, 20, 1], [15, 10, 22, 1], [32, 25, 22, 1], [18, 10, 22, 2], [15, 23, 15, 2], [16, 10, 23, 0], [14, 11, 25, 1], [8, 18, 12, 2], [11, 12, 25, 0], [0, 15, 14, 1], [15, 13, 23, 1], [31, 15, 15, 2], [20, 21, 18, 2], [17, 24, 16, 2], [26, 24, 19, 1], [17, 23, 15, 1], [22, 12, 19, 2], [10, 14, 22, 2], [6, 18, 13, 2], [22, 24, 15, 1], [25, 24, 18, 2], [11, 11, 25, 0], [20, 23, 13, 2], [15, 11, 25, 2], [25, 11, 18, 1], [28, 20, 10, 1], [28, 24, 19, 0], [13, 10, 25, 0], [16, 21, 17, 2], [21, 23, 17, 1], [15, 12, 25, 2], [19, 10, 19, 2], [22, 10, 18, 0], [5, 18, 15, 0], [23, 11, 16, 0], [34, 24, 22, 1], [7, 20, 15, 2], [28, 26, 19, 0], [19, 13, 12, 0], [28, 25, 21, 2], [23, 25, 15, 2], [17, 12, 24, 1], [19, 8, 19, 0], [18, 22, 14, 2], [14, 21, 16, 1], [11, 13, 24, 1], [24, 25, 18, 2], [8, 18, 14, 1], [17, 9, 20, 0], [6, 19, 17, 2], [20, 10, 22, 1], [11, 20, 14, 2], [7, 19, 16, 1], [13, 13, 25, 2], [14, 11, 25, 2], [17, 9, 24, 2], [19, 23, 16, 2], [22, 16, 12, 2], [16, 11, 23, 0], [17, 11, 23, 1], [6, 18, 15, 0], [14, 20, 15, 1], [23, 15, 10, 2], [32, 17, 12, 0], [20, 11, 18, 1], [15, 21, 15, 2], [5, 19, 14, 1], [12, 22, 15, 2], [24, 17, 9, 1], [6, 18, 14, 2], [31, 24, 21, 2], [16, 10, 22, 0], [15, 9, 23, 0], [16, 11, 12, 1], [16, 9, 24, 1], [10, 14, 26, 0], [10, 14, 24, 1], [13, 20, 12, 2], [21, 25, 18, 2], [19, 24, 16, 2], [21, 12, 18, 0], [22, 24, 16, 1], [14, 12, 25, 2], [22, 23, 17, 1], [18, 23, 16, 1], [4, 17, 15, 0], [7, 18, 15, 2], [0, 17, 16, 2], [10, 20, 13, 2], [19, 11, 10, 2], [36, 24, 24, 2], [16, 12, 22, 0], [24, 24, 19, 2], [3, 17, 15, 2], [34, 25, 22, 2], [26, 23, 18, 1], [23, 12, 19, 1], [14, 11, 26, 1], [21, 25, 17, 1], [35, 24, 23, 2], [17, 12, 23, 1], [15, 11, 24, 0], [16, 11, 13, 1], [7, 17, 13, 1], [22, 24, 16, 2], [13, 20, 14, 1], [17, 9, 21, 0], [3, 18, 15, 0], [32, 24, 23, 2], [18, 11, 23, 2], [22, 15, 12, 2], [25, 12, 17, 2], [14, 12, 24, 0], [15, 11, 27, 2], [7, 20, 15, 1], [14, 11, 22, 0], [23, 25, 19, 2], [27, 24, 19, 2], [17, 12, 21, 0], [20, 25, 16, 1], [10, 9, 18, 2], [17, 9, 23, 1], [36, 24, 23, 2], [15, 10, 14, 0], [15, 11, 23, 0], [8, 19, 15, 2], [21, 11, 19, 2], [9, 20, 14, 2], [12, 12, 27, 1], [24, 11, 19, 2], [15, 11, 26, 1], [5, 18, 16, 2], [27, 25, 22, 2], [17, 22, 14, 1], [21, 24, 16, 2], [0, 16, 15, 2], [16, 21, 13, 1], [1, 16, 15, 2], [19, 10, 20, 2], [22, 11, 18, 2], [29, 25, 22, 2]])
      heatmap = Heatmap(model=grasp_model, a_space=np.linspace(-np.pi / 2 + 0.1, np.pi / 2 - 0.1, 37))
      heatmap_image = heatmap.render(image_grasp, reward_index=0, save_path=save_path + '/heatmap-grasp.png', indices=indices)

      heatmap = None
      del heatmap


      indices = np.array([[22, 43, 21, 0], [17, 58, 20, 0], [29, 62, 41, 0], [19, 38, 21, 0], [16, 55, 18, 0], [9, 29, 28, 0], [30, 68, 42, 0], [17, 23, 49, 0], [12, 47, 15, 0], [16, 37, 22, 0], [13, 48, 15, 0], [35, 52, 34, 0], [9, 39, 13, 0], [17, 27, 48, 0], [17, 26, 49, 0], [12, 51, 68, 0], [25, 49, 23, 0], [18, 61, 23, 0], [30, 59, 41, 0], [36, 28, 21, 0], [11, 45, 15, 0], [30, 66, 44, 0], [17, 21, 48, 0], [32, 3, 45, 0], [17, 39, 21, 0], [21, 40, 21, 0], [16, 40, 70, 0], [14, 45, 69, 0], [10, 43, 15, 0], [17, 19, 48, 0], [16, 55, 19, 0], [35, 26, 23, 0], [20, 62, 25, 0], [11, 44, 14, 0], [17, 22, 48, 0], [36, 32, 22, 0], [18, 40, 21, 0], [29, 64, 41, 0], [29, 64, 39, 0], [14, 51, 16, 0], [21, 45, 21, 0], [17, 60, 20, 0], [19, 40, 21, 0], [17, 25, 49, 0], [14, 50, 17, 0], [10, 42, 14, 0], [18, 46, 21, 0], [29, 63, 41, 0], [10, 40, 14, 0], [13, 49, 16, 0], [30, 60, 42, 0], [22, 42, 21, 0], [29, 58, 39, 0], [35, 28, 22, 0], [16, 36, 22, 0], [36, 30, 21, 0], [36, 12, 38, 0], [11, 52, 66, 0], [17, 23, 48, 0], [13, 50, 16, 0], [36, 54, 36, 0], [10, 45, 14, 0], [7, 57, 62, 0], [30, 61, 41, 0], [30, 61, 42, 0], [0, 22, 20, 0], [1, 67, 50, 0], [17, 37, 21, 0], [30, 67, 44, 0], [36, 21, 22, 0], [36, 27, 22, 0], [10, 42, 13, 0], [30, 64, 41, 0], [29, 56, 27, 0], [29, 67, 44, 0], [17, 43, 22, 0], [17, 30, 49, 0], [9, 56, 64, 0], [29, 14, 57, 0], [17, 28, 49, 0], [0, 13, 38, 0], [28, 66, 39, 0], [15, 31, 22, 0], [17, 27, 50, 0], [22, 44, 22, 0], [15, 34, 22, 0], [15, 40, 69, 0], [12, 44, 15, 0], [18, 32, 69, 0], [17, 57, 21, 0], [29, 64, 40, 0], [36, 31, 21, 0], [18, 35, 21, 0], [11, 36, 14, 0], [11, 43, 15, 0], [34, 55, 33, 0], [18, 39, 21, 0], [15, 35, 23, 0], [17, 42, 21, 0], [36, 10, 38, 0], [15, 35, 22, 0], [34, 54, 33, 0], [29, 4, 53, 0], [10, 38, 14, 0], [35, 50, 34, 0], [36, 11, 38, 0], [10, 40, 13, 0], [30, 62, 43, 0], [20, 61, 26, 0], [18, 59, 22, 0], [12, 48, 15, 0], [1, 22, 34, 0], [17, 39, 22, 0], [17, 35, 69, 0], [36, 27, 21, 0], [30, 61, 43, 0], [17, 27, 49, 0], [11, 41, 14, 0], [9, 56, 65, 0], [9, 54, 65, 0], [11, 45, 14, 0], [16, 39, 21, 0], [17, 35, 21, 0], [10, 43, 12, 0], [13, 43, 68, 0], [16, 56, 20, 0], [19, 39, 21, 0], [28, 67, 38, 0], [17, 25, 48, 0], [31, 67, 45, 0], [30, 67, 42, 0], [35, 26, 22, 0], [11, 31, 26, 0], [13, 52, 16, 0], [13, 52, 15, 0], [15, 30, 23, 0], [10, 39, 14, 0], [29, 67, 40, 0], [16, 58, 20, 0], [17, 36, 69, 0], [15, 37, 22, 0], [13, 46, 68, 0], [29, 71, 40, 0], [29, 61, 40, 0], [17, 40, 21, 0], [21, 51, 23, 0], [25, 48, 23, 0], [36, 23, 21, 0], [0, 25, 39, 0], [14, 49, 17, 0], [16, 53, 19, 0], [29, 58, 41, 0], [17, 56, 20, 0], [29, 59, 41, 0], [17, 20, 49, 0], [8, 26, 29, 0], [30, 62, 42, 0], [36, 30, 22, 0], [9, 38, 13, 0], [30, 66, 42, 0], [31, 63, 44, 0], [35, 27, 23, 0], [29, 67, 41, 0], [16, 53, 18, 0], [29, 61, 41, 0], [29, 66, 40, 0], [9, 29, 29, 0], [30, 63, 43, 0], [9, 30, 28, 0], [10, 45, 12, 0], [35, 25, 22, 0], [17, 24, 48, 0], [36, 33, 21, 0], [17, 45, 21, 0], [14, 54, 17, 0], [30, 62, 41, 0], [17, 37, 70, 0], [17, 37, 22, 0], [10, 43, 14, 0], [36, 53, 36, 0], [22, 43, 22, 0], [18, 38, 21, 0], [16, 38, 22, 0], [30, 6, 51, 0], [9, 58, 65, 0], [16, 26, 50, 0], [29, 63, 40, 0], [29, 65, 40, 0], [11, 43, 14, 0], [34, 50, 33, 0], [30, 65, 43, 0], [29, 62, 40, 0], [20, 46, 23, 0], [10, 46, 14, 0], [17, 41, 21, 0], [20, 29, 69, 0], [20, 50, 23, 0], [30, 4, 51, 0], [0, 66, 46, 0], [30, 63, 41, 0], [14, 52, 17, 0], [17, 34, 21, 0], [29, 8, 53, 0], [29, 65, 39, 0], [10, 37, 13, 0], [29, 61, 39, 0], [19, 61, 23, 0], [36, 60, 35, 0], [19, 41, 21, 0], [16, 35, 22, 0], [30, 64, 42, 0], [17, 38, 69, 0], [10, 38, 13, 0], [25, 15, 62, 0], [36, 25, 21, 0], [29, 58, 37, 0], [36, 26, 22, 0], [25, 50, 23, 0], [21, 41, 21, 0], [0, 16, 39, 0], [16, 41, 69, 0], [17, 35, 22, 0], [17, 58, 21, 0], [30, 66, 43, 0], [13, 48, 68, 0], [16, 54, 19, 0], [15, 52, 18, 0], [33, 50, 33, 0], [29, 60, 41, 0], [17, 59, 21, 0], [29, 66, 42, 0], [30, 63, 42, 0], [18, 58, 23, 0], [10, 41, 13, 0], [17, 43, 21, 0], [1, 26, 34, 0], [29, 12, 57, 0], [30, 65, 42, 0], [22, 47, 22, 0], [16, 38, 21, 0], [10, 44, 15, 0], [36, 22, 21, 0], [9, 53, 65, 0], [14, 53, 17, 0], [29, 68, 39, 0], [19, 61, 24, 0], [10, 41, 14, 0], [29, 68, 40, 0], [21, 62, 27, 0], [29, 69, 41, 0], [18, 41, 21, 0], [35, 50, 35, 0], [28, 59, 38, 0], [28, 68, 39, 0], [16, 27, 50, 0], [15, 54, 18, 0], [8, 27, 29, 0], [12, 48, 68, 0], [30, 64, 43, 0], [36, 21, 21, 0], [29, 60, 40, 0], [35, 28, 23, 0], [36, 7, 38, 0], [36, 25, 23, 0], [36, 31, 22, 0], [29, 70, 40, 0], [30, 60, 43, 0], [17, 41, 22, 0], [36, 34, 21, 0], [15, 50, 18, 0], [9, 39, 14, 0], [19, 61, 25, 0], [36, 29, 22, 0], [12, 49, 68, 0], [12, 43, 14, 0], [10, 39, 15, 0], [17, 20, 48, 0], [9, 57, 65, 0], [20, 48, 22, 0], [28, 63, 38, 0], [30, 58, 42, 0], [36, 49, 35, 0], [32, 1, 45, 0], [17, 24, 49, 0], [8, 56, 64, 0], [14, 50, 16, 0], [36, 22, 22, 0], [30, 60, 41, 0], [13, 48, 16, 0], [17, 29, 48, 0], [35, 51, 34, 0], [20, 63, 26, 0], [20, 47, 22, 0], [36, 20, 22, 0], [18, 34, 69, 0], [34, 51, 34, 0], [17, 29, 49, 0], [11, 48, 66, 0], [1, 25, 34, 0], [15, 55, 18, 0], [28, 68, 38, 0], [36, 26, 21, 0], [12, 45, 15, 0], [1, 14, 37, 0], [19, 45, 21, 0], [20, 30, 69, 0], [31, 59, 44, 0], [18, 37, 21, 0], [14, 42, 69, 0], [30, 69, 43, 0], [10, 41, 15, 0], [10, 44, 13, 0], [14, 54, 16, 0], [16, 40, 21, 0], [18, 42, 22, 0], [11, 30, 26, 0], [29, 59, 40, 0], [19, 60, 23, 0], [11, 50, 67, 0], [17, 28, 48, 0], [10, 37, 14, 0], [14, 31, 23, 0], [16, 39, 70, 0], [28, 64, 38, 0], [18, 57, 23, 0], [13, 30, 24, 0], [13, 53, 15, 0], [28, 62, 38, 0], [10, 42, 15, 0], [11, 39, 14, 0], [13, 44, 68, 0], [15, 51, 18, 0], [16, 33, 22, 0], [36, 19, 22, 0], [35, 29, 23, 0], [10, 39, 13, 0], [11, 48, 67, 0], [18, 56, 23, 0], [36, 24, 22, 0], [29, 66, 41, 0], [28, 61, 38, 0], [17, 26, 48, 0], [9, 55, 65, 0], [12, 46, 14, 0], [21, 63, 27, 0], [34, 16, 42, 0], [36, 52, 36, 0], [17, 60, 21, 0], [8, 38, 13, 0], [20, 45, 22, 0], [36, 64, 35, 0], [12, 50, 68, 0], [18, 38, 22, 0], [17, 57, 20, 0], [23, 48, 22, 0], [9, 27, 28, 0], [29, 65, 42, 0], [17, 36, 21, 0], [11, 49, 66, 0], [36, 25, 22, 0], [36, 51, 36, 0], [16, 28, 50, 0], [18, 35, 69, 0], [16, 25, 50, 0], [21, 44, 21, 0], [36, 8, 38, 0], [29, 62, 42, 0], [30, 59, 42, 0], [30, 68, 43, 0], [15, 55, 17, 0], [18, 56, 22, 0], [20, 60, 25, 0], [17, 59, 20, 0], [29, 60, 39, 0], [30, 65, 41, 0], [35, 54, 34, 0], [9, 40, 14, 0], [10, 45, 13, 0], [12, 49, 15, 0], [21, 64, 27, 0], [21, 29, 68, 0], [35, 55, 34, 0], [16, 21, 50, 0], [9, 55, 64, 0], [17, 21, 49, 0], [21, 50, 23, 0], [1, 27, 34, 0], [11, 38, 14, 0], [35, 23, 23, 0], [30, 67, 43, 0], [35, 65, 33, 0], [17, 29, 50, 0], [17, 24, 50, 0], [30, 66, 41, 0], [17, 36, 22, 0], [17, 40, 22, 0], [29, 61, 38, 0], [16, 40, 69, 0], [15, 33, 22, 0], [7, 57, 61, 0], [36, 9, 38, 0], [16, 24, 50, 0], [10, 48, 14, 0], [15, 33, 23, 0], [27, 8, 58, 0], [8, 54, 64, 0], [9, 42, 13, 0], [29, 69, 40, 0], [11, 52, 67, 0], [7, 56, 62, 0], [25, 12, 63, 0], [14, 48, 17, 0], [14, 51, 17, 0], [16, 23, 50, 0], [36, 50, 36, 0], [29, 58, 40, 0], [16, 57, 20, 0], [35, 29, 22, 0], [30, 65, 44, 0], [20, 50, 22, 0], [17, 39, 70, 0], [36, 61, 55, 0], [10, 39, 16, 0], [29, 65, 41, 0], [10, 31, 27, 0], [11, 50, 66, 0], [29, 55, 27, 0], [20, 64, 26, 0], [28, 62, 39, 0], [13, 51, 16, 0], [18, 58, 21, 0], [20, 59, 25, 0], [29, 64, 42, 0], [10, 47, 14, 0], [21, 45, 22, 0], [11, 42, 14, 0], [17, 34, 22, 0], [15, 43, 69, 0], [9, 37, 13, 0], [14, 49, 16, 0], [7, 55, 62, 0], [9, 53, 64, 0], [18, 38, 20, 0], [19, 42, 22, 0], [15, 53, 18, 0], [1, 16, 37, 0], [35, 4, 37, 0], [29, 62, 39, 0], [14, 48, 16, 0], [12, 48, 14, 0], [9, 40, 13, 0], [30, 7, 51, 0], [8, 57, 64, 0], [20, 42, 20, 0], [12, 48, 67, 0], [34, 14, 43, 0], [10, 35, 15, 0], [35, 1, 37, 0], [24, 15, 64, 0], [10, 46, 13, 0], [9, 51, 64, 0], [10, 43, 13, 0], [12, 46, 15, 0], [11, 41, 15, 0], [9, 51, 65, 0], [19, 63, 24, 0], [17, 37, 69, 0], [20, 42, 21, 0], [29, 15, 57, 0], [10, 44, 14, 0], [17, 33, 70, 0], [29, 63, 39, 0], [11, 51, 67, 0], [19, 32, 69, 0], [16, 34, 22, 0], [36, 20, 21, 0], [18, 33, 69, 0], [11, 48, 14, 0], [17, 55, 21, 0], [36, 23, 22, 0], [23, 53, 26, 0], [12, 43, 15, 0], [6, 34, 14, 0], [15, 39, 69, 0], [7, 54, 62, 0], [20, 40, 21, 0], [17, 44, 21, 0], [16, 55, 20, 0], [31, 64, 44, 0], [15, 32, 22, 0], [27, 51, 24, 0], [29, 67, 39, 0], [36, 24, 21, 0], [18, 36, 21, 0], [28, 65, 38, 0], [36, 29, 21, 0], [33, 52, 33, 0], [16, 56, 19, 0], [36, 52, 35, 0], [14, 32, 23, 0]])
      heatmap = Heatmap(model=place_model, a_space=np.linspace(-np.pi / 2 + 0.1, np.pi / 2 - 0.1, 37))
      heatmap_image = heatmap.render(image_place, image_goal, save_path=save_path + '/heatmap-place.png', indices=indices)

   elif model_type == "grasp":
      save_path = "./data/heatmaps"
      device = "cuda" if torch.cuda.is_available() else "cpu"

      image_grasp = cv2.imread("./data/goal/rec_goal1.png", cv2.IMREAD_GRAYSCALE)


      grasp_model = GraspModel(1).float().to(device)
      cptfile = './data/checkpoints/grasp_model.cpt'
      cpt = torch.load(cptfile)
      grasp_model.load_state_dict(cpt['grasp_model_state_dict'])

      indices = np.array([[22, 25, 18, 1], [13, 21, 17, 2], [7, 19, 16, 2], [11, 22, 17, 2], [15, 10, 12, 2], [20, 24, 15, 2], [17, 23, 15, 0], [0, 16, 15, 1], [15, 22, 15, 0], [6, 17, 16, 1], [20, 10, 20, 1], [32, 15, 12, 0], [28, 26, 21, 1], [30, 26, 20, 1], [14, 12, 24, 1], [18, 11, 21, 1], [12, 11, 24, 0], [9, 14, 26, 1], [19, 10, 19, 0], [11, 12, 26, 0], [22, 24, 19, 2], [4, 16, 17, 2], [12, 11, 25, 1], [23, 12, 17, 0], [28, 19, 10, 2], [22, 25, 17, 2], [18, 11, 23, 1], [15, 11, 25, 1], [22, 24, 17, 1], [14, 23, 15, 1], [0, 15, 16, 2], [13, 20, 17, 2], [9, 19, 14, 1], [18, 9, 20, 1], [15, 12, 24, 2], [2, 17, 13, 1], [16, 10, 23, 2], [16, 10, 21, 0], [16, 23, 14, 2], [4, 18, 16, 2], [9, 18, 16, 2], [25, 17, 9, 1], [9, 19, 16, 2], [12, 9, 16, 1], [16, 11, 24, 1], [23, 10, 18, 1], [22, 11, 18, 0], [19, 9, 20, 0], [3, 17, 16, 1], [7, 19, 15, 1], [9, 20, 16, 2], [28, 13, 14, 0], [11, 20, 16, 1], [24, 25, 16, 2], [14, 12, 23, 0], [24, 10, 18, 1], [2, 15, 16, 2], [15, 9, 22, 0], [13, 12, 24, 2], [14, 21, 14, 1], [15, 10, 24, 0], [0, 16, 16, 1], [4, 17, 16, 2], [27, 25, 19, 0], [20, 10, 19, 1], [21, 11, 17, 0], [17, 12, 22, 0], [2, 15, 16, 1], [12, 14, 25, 0], [9, 20, 17, 2], [14, 21, 15, 0], [18, 9, 23, 1], [12, 13, 23, 2], [22, 10, 18, 1], [14, 11, 26, 2], [31, 25, 21, 1], [12, 20, 14, 2], [17, 9, 22, 2], [26, 24, 18, 1], [16, 13, 21, 0], [17, 22, 16, 2], [17, 22, 18, 2], [18, 23, 16, 0], [20, 12, 19, 0], [32, 26, 21, 1], [35, 23, 22, 1], [11, 19, 15, 2], [16, 11, 21, 2], [4, 18, 15, 2], [12, 21, 16, 2], [15, 11, 21, 0], [2, 16, 16, 2], [14, 10, 24, 1], [4, 16, 15, 1], [18, 11, 19, 0], [24, 17, 9, 2], [9, 20, 15, 2], [18, 11, 21, 0], [0, 15, 15, 2], [8, 20, 13, 1], [16, 9, 21, 0], [0, 15, 17, 2], [8, 9, 20, 0], [15, 13, 24, 2], [19, 22, 18, 2], [16, 22, 15, 1], [12, 12, 24, 2], [21, 11, 21, 1], [1, 16, 16, 1], [9, 17, 12, 2], [14, 20, 12, 2], [25, 26, 21, 2], [6, 17, 15, 1], [31, 25, 23, 2], [17, 11, 20, 0], [22, 25, 18, 2], [17, 12, 23, 2], [7, 20, 13, 1], [15, 9, 24, 1], [13, 20, 16, 1], [1, 16, 14, 2], [19, 12, 22, 1], [7, 15, 27, 0], [17, 10, 24, 1], [18, 9, 24, 2], [19, 24, 16, 0], [14, 12, 23, 1], [17, 11, 22, 1], [18, 10, 20, 1], [8, 20, 14, 1], [19, 12, 21, 1], [10, 20, 15, 1], [19, 10, 20, 1], [26, 23, 17, 1], [9, 14, 24, 2], [14, 22, 14, 1], [13, 12, 25, 0], [19, 25, 19, 2], [11, 19, 14, 2], [17, 10, 22, 0], [13, 13, 25, 0], [8, 15, 27, 0], [15, 11, 23, 1], [33, 17, 12, 0], [14, 10, 23, 0], [20, 9, 20, 1], [13, 13, 24, 0], [12, 20, 16, 1], [7, 16, 13, 1], [13, 13, 26, 1], [15, 12, 24, 0], [22, 24, 20, 2], [18, 11, 22, 1], [6, 16, 14, 1], [32, 25, 22, 2], [17, 22, 17, 1], [14, 13, 24, 1], [34, 24, 21, 1], [9, 19, 17, 2], [3, 17, 14, 1], [25, 17, 9, 0], [24, 24, 18, 2], [16, 11, 25, 2], [27, 13, 16, 2], [1, 17, 16, 2], [17, 22, 16, 1], [10, 13, 26, 0], [30, 26, 22, 2], [24, 24, 17, 2], [20, 11, 22, 1], [2, 16, 18, 2], [20, 10, 18, 0], [26, 25, 18, 1], [16, 11, 21, 1], [12, 21, 14, 0], [10, 14, 25, 0], [24, 11, 17, 1], [13, 11, 25, 1], [13, 12, 27, 1], [27, 26, 21, 1], [18, 10, 19, 2], [1, 16, 16, 2], [16, 10, 25, 2], [26, 25, 19, 1], [9, 15, 25, 0], [12, 9, 16, 2], [2, 16, 13, 1], [9, 18, 13, 2], [21, 24, 18, 1], [16, 23, 14, 1], [18, 11, 22, 2], [29, 25, 21, 1], [11, 12, 24, 0], [20, 22, 16, 2], [27, 26, 19, 2], [24, 12, 16, 1], [11, 15, 24, 0], [33, 23, 22, 1], [16, 22, 15, 2], [9, 15, 27, 0], [27, 26, 21, 2], [22, 10, 20, 1], [25, 26, 19, 1], [10, 18, 15, 2], [35, 24, 23, 1], [20, 9, 21, 2], [24, 12, 18, 1], [13, 20, 15, 1], [20, 23, 17, 1], [22, 16, 8, 1], [13, 11, 26, 1], [30, 25, 21, 2], [9, 20, 14, 0], [7, 15, 28, 0], [13, 13, 24, 1], [10, 21, 13, 1], [23, 26, 17, 0], [19, 23, 16, 1], [3, 17, 15, 0], [7, 19, 15, 2], [14, 20, 13, 2], [23, 25, 17, 1], [1, 15, 13, 2], [20, 10, 19, 0], [18, 9, 21, 1], [20, 12, 21, 1], [20, 10, 21, 1], [18, 10, 23, 2], [16, 12, 26, 2], [29, 26, 18, 2], [36, 24, 24, 1], [22, 15, 10, 1], [18, 23, 15, 1], [2, 15, 17, 2], [14, 20, 16, 2], [32, 16, 14, 1], [17, 11, 11, 2], [21, 11, 18, 2], [24, 13, 17, 1], [3, 16, 14, 1], [19, 9, 20, 1], [11, 12, 23, 1], [35, 25, 23, 2], [15, 12, 22, 0], [34, 24, 21, 2], [23, 22, 16, 1], [14, 20, 14, 2], [21, 24, 15, 1], [9, 18, 15, 2], [10, 20, 14, 2], [9, 13, 23, 2], [17, 11, 23, 2], [10, 18, 16, 1], [21, 25, 18, 1], [6, 16, 15, 2], [8, 19, 16, 1], [30, 25, 20, 2], [24, 25, 18, 0], [12, 11, 26, 0], [16, 11, 21, 0], [23, 23, 20, 2], [18, 11, 20, 0], [11, 21, 15, 1], [19, 9, 21, 2], [10, 19, 16, 2], [8, 18, 15, 2], [15, 22, 15, 2], [11, 21, 16, 2], [29, 25, 19, 2], [13, 11, 26, 2], [18, 23, 14, 1], [10, 21, 15, 2], [29, 15, 16, 2], [9, 19, 15, 2], [6, 17, 14, 1], [28, 12, 17, 2], [23, 24, 17, 1], [18, 12, 23, 2], [31, 24, 21, 1], [15, 11, 24, 1], [14, 12, 22, 0], [12, 14, 24, 0], [20, 24, 16, 1], [24, 11, 17, 0], [12, 20, 14, 1], [12, 13, 24, 0], [0, 15, 17, 1], [11, 14, 25, 0], [7, 20, 14, 2], [23, 11, 18, 1], [34, 24, 24, 2], [0, 16, 17, 2], [24, 12, 16, 0], [5, 16, 15, 1], [13, 19, 16, 2], [36, 25, 25, 2], [13, 11, 27, 1], [18, 12, 21, 1], [10, 13, 25, 0], [26, 18, 9, 0], [36, 23, 22, 1], [31, 24, 21, 0], [15, 12, 24, 1], [13, 12, 23, 0], [24, 12, 17, 1], [15, 10, 22, 0], [9, 21, 14, 2], [16, 23, 17, 2], [30, 14, 13, 0], [20, 11, 19, 1], [2, 16, 15, 1], [19, 9, 21, 1], [8, 15, 25, 0], [11, 13, 25, 0], [34, 24, 22, 2], [6, 19, 14, 1], [24, 23, 18, 2], [25, 18, 9, 2], [21, 15, 10, 0], [8, 19, 13, 1], [24, 26, 19, 1], [2, 15, 15, 1], [17, 11, 23, 0], [31, 25, 20, 1], [9, 13, 26, 0], [24, 16, 9, 1], [11, 22, 14, 1], [5, 18, 15, 2], [20, 12, 21, 2], [12, 12, 26, 0], [10, 13, 27, 0], [23, 16, 8, 1], [20, 9, 20, 0], [1, 17, 18, 2], [21, 24, 15, 2], [11, 19, 14, 1], [4, 17, 17, 2], [31, 24, 22, 2], [26, 17, 9, 1], [26, 13, 15, 0], [16, 11, 23, 2], [23, 16, 10, 2], [23, 11, 19, 1], [14, 11, 24, 1], [13, 12, 25, 2], [19, 11, 22, 1], [13, 20, 15, 2], [33, 25, 23, 2], [34, 25, 22, 1], [10, 13, 24, 2], [35, 19, 11, 0], [13, 13, 24, 2], [11, 20, 15, 2], [32, 25, 21, 1], [15, 11, 23, 2], [8, 19, 12, 2], [24, 26, 19, 2], [5, 17, 12, 2], [30, 25, 22, 2], [28, 24, 20, 1], [15, 10, 22, 1], [32, 25, 22, 1], [18, 10, 22, 2], [15, 23, 15, 2], [16, 10, 23, 0], [14, 11, 25, 1], [8, 18, 12, 2], [11, 12, 25, 0], [0, 15, 14, 1], [15, 13, 23, 1], [31, 15, 15, 2], [20, 21, 18, 2], [17, 24, 16, 2], [26, 24, 19, 1], [17, 23, 15, 1], [22, 12, 19, 2], [10, 14, 22, 2], [6, 18, 13, 2], [22, 24, 15, 1], [25, 24, 18, 2], [11, 11, 25, 0], [20, 23, 13, 2], [15, 11, 25, 2], [25, 11, 18, 1], [28, 20, 10, 1], [28, 24, 19, 0], [13, 10, 25, 0], [16, 21, 17, 2], [21, 23, 17, 1], [15, 12, 25, 2], [19, 10, 19, 2], [22, 10, 18, 0], [5, 18, 15, 0], [23, 11, 16, 0], [34, 24, 22, 1], [7, 20, 15, 2], [28, 26, 19, 0], [19, 13, 12, 0], [28, 25, 21, 2], [23, 25, 15, 2], [17, 12, 24, 1], [19, 8, 19, 0], [18, 22, 14, 2], [14, 21, 16, 1], [11, 13, 24, 1], [24, 25, 18, 2], [8, 18, 14, 1], [17, 9, 20, 0], [6, 19, 17, 2], [20, 10, 22, 1], [11, 20, 14, 2], [7, 19, 16, 1], [13, 13, 25, 2], [14, 11, 25, 2], [17, 9, 24, 2], [19, 23, 16, 2], [22, 16, 12, 2], [16, 11, 23, 0], [17, 11, 23, 1], [6, 18, 15, 0], [14, 20, 15, 1], [23, 15, 10, 2], [32, 17, 12, 0], [20, 11, 18, 1], [15, 21, 15, 2], [5, 19, 14, 1], [12, 22, 15, 2], [24, 17, 9, 1], [6, 18, 14, 2], [31, 24, 21, 2], [16, 10, 22, 0], [15, 9, 23, 0], [16, 11, 12, 1], [16, 9, 24, 1], [10, 14, 26, 0], [10, 14, 24, 1], [13, 20, 12, 2], [21, 25, 18, 2], [19, 24, 16, 2], [21, 12, 18, 0], [22, 24, 16, 1], [14, 12, 25, 2], [22, 23, 17, 1], [18, 23, 16, 1], [4, 17, 15, 0], [7, 18, 15, 2], [0, 17, 16, 2], [10, 20, 13, 2], [19, 11, 10, 2], [36, 24, 24, 2], [16, 12, 22, 0], [24, 24, 19, 2], [3, 17, 15, 2], [34, 25, 22, 2], [26, 23, 18, 1], [23, 12, 19, 1], [14, 11, 26, 1], [21, 25, 17, 1], [35, 24, 23, 2], [17, 12, 23, 1], [15, 11, 24, 0], [16, 11, 13, 1], [7, 17, 13, 1], [22, 24, 16, 2], [13, 20, 14, 1], [17, 9, 21, 0], [3, 18, 15, 0], [32, 24, 23, 2], [18, 11, 23, 2], [22, 15, 12, 2], [25, 12, 17, 2], [14, 12, 24, 0], [15, 11, 27, 2], [7, 20, 15, 1], [14, 11, 22, 0], [23, 25, 19, 2], [27, 24, 19, 2], [17, 12, 21, 0], [20, 25, 16, 1], [10, 9, 18, 2], [17, 9, 23, 1], [36, 24, 23, 2], [15, 10, 14, 0], [15, 11, 23, 0], [8, 19, 15, 2], [21, 11, 19, 2], [9, 20, 14, 2], [12, 12, 27, 1], [24, 11, 19, 2], [15, 11, 26, 1], [5, 18, 16, 2], [27, 25, 22, 2], [17, 22, 14, 1], [21, 24, 16, 2], [0, 16, 15, 2], [16, 21, 13, 1], [1, 16, 15, 2], [19, 10, 20, 2], [22, 11, 18, 2], [29, 25, 22, 2]])
      heatmap = Heatmap(model=grasp_model, a_space=np.linspace(-np.pi / 2 + 0.1, np.pi / 2 - 0.1, 37))
      heatmap_image = heatmap.render(image_grasp, save_path=save_path + '/heatmap-grasp.png', indices=indices)
