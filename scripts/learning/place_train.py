#!/usr/bin/python3
import datetime
import time
import json
import pickle
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.models import GraspModel, PlaceModel, MergeModel, Combined_model
from learning.place_datasets import CustomDataset
from learning.metrics import Losses


class Train:
   def __init__(self, dataset_path=None, image_format='png'):
      self.input_shape = [None, None, 1] if True else [None, None, 3]
      self.z_shape = 48
      self.train_batch_size = 512
      self.validation_batch_size = 512
      self.percent_validation_set = 0.2
      self.device = "cuda" if torch.cuda.is_available() else "cpu"
      torch.manual_seed(0)

      self.previous_epoch = 0
      self.dataset_path = dataset_path

      self.dataset_tensor = []

   def run(self, load_model=True):
      self.writer = SummaryWriter(log_dir="./data/logs")
      # get dataset
      with open(self.dataset_path, mode="rt", encoding="utf-8") as f:
         all_data = json.load(f)
      custom_ds = CustomDataset(all_data, seed=42)
      datasets = custom_ds.get_data(self.dataset_tensor)
      datasets_length = len(datasets)
      val_data_length = int(datasets_length * self.percent_validation_set)
      train_data_length = datasets_length - val_data_length
      train_dataset, val_dataset = torch.utils.data.random_split(datasets, [train_data_length, val_data_length])

      train_dataloaders = DataLoader(train_dataset, 
                                    batch_size=self.train_batch_size,
                                    shuffle=True,
                                    num_workers=0, 
                                    drop_last=False,
                                    pin_memory=True
                                    )

      val_dataloader = DataLoader(val_dataset, 
                                 batch_size=self.validation_batch_size,
                                 shuffle=True,
                                 num_workers=0,
                                 drop_last=False,
                                 pin_memory=True)
                                 
      self.dataset_tensor = datasets

      # set up nn model
      model = PlaceModel(self.input_shape[2]*2).to(self.device)

      # optimizer
      reward_param = []
      z_param = []
      # merge_param = []
      other_param = []
      for name, param in model.named_parameters():
         if '_r_last' in name:
            reward_param.append(param)
         elif '_z_last' in name:
            z_param.append(param)
         # elif 'merge_model.linear_block' in name:
         #    merge_param.append(param)
         else:
            other_param.append(param)
      optimizer = torch.optim.Adam([
         {'params': reward_param, 'weight_decay': 0.0},
         {'params': z_param, 'weight_decay': 0.0005},
         {'params': other_param, 'weight_decay': 0.001},
      ], lr=1e-4)

      # loss function
      criterion = torch.nn.BCELoss()

      # load model
      if load_model:
         cptfile = './data/checkpoints/place_model.cpt'
         cpt = torch.load(cptfile)
         stdict_m = cpt['place_model_state_dict']
         stdict_o = cpt['opt_state_dict']
         model.load_state_dict(stdict_m)
         optimizer.load_state_dict(stdict_o)

      epoch = 10000
      self.current_e = 0
      with tqdm(range(epoch)) as pbar_epoch:
         for e in pbar_epoch:
            self.train(train_dataloaders, model, optimizer)
            self.current_e += 1

      self.writer.close()
      self.test(val_dataloader, model, criterion, optimizer)

      outfile = './data/checkpoints/place_model.cpt'
      torch.save({'place_model_state_dict': model.state_dict(),
                  'opt_state_dict': optimizer.state_dict(),
                  }, outfile)




   def train(self, dataloader, model, optimizer):
      train_loss= 0
      size = len(dataloader.dataset)
      for x, y in dataloader:
         if x[0].shape[0] == 1:
            break
         x = tuple(torch.reshape(x_arr, (-1, 1, 32, 32)).to(self.device) for x_arr in x)
         y = tuple(torch.reshape(y_arr, (-1, 1, 1, 1)).to(self.device) for y_arr in y)

         z_p, reward_p= model(x[0],x[1])
         loss_fn = torch.nn.BCELoss(weight = y[1])
         loss = loss_fn(reward_p, y[0])

         # Backpropagation
         optimizer.zero_grad()
         loss.backward()
         optimizer.step()
         train_loss += loss.item()
      self.writer.add_scalar("loss", train_loss/size, self.current_e)




   def test(self, dataloader, model, loss_fn, optimizer):
      size = len(dataloader.dataset)
      model.eval()
      test_loss, correct = 0, 0
      with torch.no_grad():
         for x, y in dataloader:
            if x[0].shape[0] == 1:
               break
            x = tuple(torch.reshape(x_arr, (-1, 1, 32, 32)).to(self.device) for x_arr in x)
            y = tuple(torch.reshape(y_arr, (-1, 1, 1, 1)).to(self.device) for y_arr in y)

            z_p, reward_p = model(x[0],x[1])
            test_loss += loss_fn(reward_p, y[0]).item()

      test_loss /= size
      print(f"Avg loss: {test_loss:>8f}")

train = Train(
   image_format="png",
   dataset_path='./data/datasets/grasp_datasets.json'
)

train.run(False)