#!/usr/bin/env python3
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
from pathlib import Path
from sklearn.linear_model import LinearRegression
import seaborn


with open("./data/datasets/main_time.json", mode="rt", encoding="utf-8") as f:
   main_time_data = json.load(f)
with open("./data/datasets/learning_time.json", mode="rt", encoding="utf-8") as f:
   learning_time_data = json.load(f)

main_data = []
main_num = []
for t in range(len(main_time_data)):
   main_data.append(main_time_data[str(t)])
   main_num.append(t)

learning_data = []
learning_num = []
for t in range(len(learning_time_data)):
   learning_data.append(learning_time_data[str(t)])
   learning_num.append(t)

x = np.arange(10000)

###########################

main_data_np = np.array(main_data)
# print(main_data_np[:,0])
model_lr = LinearRegression()
model_lr.fit(np.array([main_num]).reshape(-1, 1), main_data_np[:,1].reshape(-1, 1))
ep = "y = " + str(model_lr.coef_[0,0]) + "x + " + str(model_lr.intercept_[0])

plt.plot(main_num, main_data_np[:,0], 'o', label="inference")
plt.plot(main_num, main_data_np[:,1], 'o', label="learning")
plt.plot(x.reshape(-1, 1), model_lr.predict(x.reshape(-1, 1)), linestyle="solid", label=ep)
plt.legend()
plt.show()

###############################

learning_data_np = np.array(learning_data)
# print(main_data_np[:,0])
model_lr1 = LinearRegression()
model_lr1.fit(np.array([learning_num]).reshape(-1, 1), learning_data_np[:,0].reshape(-1, 1))
ep1 = "y = " + str(model_lr1.coef_[0,0]) + "x + " + str(model_lr1.intercept_[0])
model_lr2 = LinearRegression()
model_lr2.fit(np.array([learning_num]).reshape(-1, 1), learning_data_np[:,1].reshape(-1, 1))
ep2 = "y = " + str(model_lr2.coef_[0,0]) + "x + " + str(model_lr2.intercept_[0])
model_lr3 = LinearRegression()
model_lr3.fit(np.array([learning_num]).reshape(-1, 1), learning_data_np[:,2].reshape(-1, 1))
ep3 = "y = " + str(model_lr3.coef_[0,0]) + "x + " + str(model_lr3.intercept_[0])

plt.plot(learning_num, learning_data_np[:,0], 'o', label="creatate_dataset")
plt.plot(learning_num, learning_data_np[:,1], 'o', label="training")
plt.plot(learning_num, learning_data_np[:,2], 'o', label="validation")
plt.plot(x.reshape(-1, 1), model_lr1.predict(x.reshape(-1, 1)), linestyle="solid", label=ep1)
plt.plot(x.reshape(-1, 1), model_lr2.predict(x.reshape(-1, 1)), linestyle="solid", label=ep2)
plt.plot(x.reshape(-1, 1), model_lr3.predict(x.reshape(-1, 1)), linestyle="solid", label=ep3)
plt.legend()
plt.show()
