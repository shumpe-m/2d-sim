#!/usr/bin/env python3
import time

import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image, ImageDraw

class Environment:
   def __init__(self, size = (752, 480), num_obj = None):
      self.rgb_size = (size[0], size[1], 3)
      self.depth_size = (size[0], size[1], 1)
      self.num_obj = num_obj

   def plot_env(self):
      rgb = np.ones(self.rgb_size) * 255
      rgb = np.clip(rgb, 0, 255).astype(np.uint8)
      depth = np.ones(self.depth_size) * 255
   
      plt.imshow(rgb)
      plt.show(block=False)
      plt.pause(2)
      plt.clf()
      plt.close()


def main():
      env = Environment(size = (752, 480), num_obj = 1)
      env.plot_env()
        


if __name__ == "__main__":
   main()