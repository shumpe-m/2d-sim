#!/usr/bin/env python3
import time
import numpy as np
import matplotlib.pyplot as plt
import pathlib

class Save:
   def __init__(self, img_path:str, data_path:str, model_path:str):
      self.img_path = img_path
      self.data_path = data_path
      self.model_path = model_path

   def save_image(self, img:np.array, file_name:str):
      path_dir = pathlib.Path(r'data/img')
      path_img = path_dir.joinpath(file_name + '.png')
      plt.savefig(path_img)

   def save_data(self, data:np.array):
       pass

   def save_model(self, model):
       pass