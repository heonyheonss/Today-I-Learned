# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 02:03:24 2023

@author: USER
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)

# %% 1. Generate data
xdata = np.array([[0,0],
                  [0,1],
                  [1,0],
                  [1,1]]).astype(np.float32)
ydata = np.array([[0],[1],[1],[0]]).astype(np.float32)

plt.scatter(xdata[:,0], xdata[:,1], color =['blue', 'red', 'red', 'blue'])
plt.show()

# %% Define
xdata = torch.tensor(xdata, dtype=torch.float32)
ydata = torch.tensor(ydata, dtype=torch.float32)

dataset = TensorDataset(xdata, ydata)
dataloader = DataLoader(dataset, batch_size=len(xdata), shuffle=True)

def model(x):
    #Logistic Regression
    hypothesis = 1 / (1 + torch.exp(torch.matmul(x, W) - b))
    return hypothesis

def loss_fn(hypothesis, labels):
    cost = -torch.mean(labels * torch.log(hypothesis) + (1 - labels) * torch.log(1 - hypothesis))
    return cost

def accuracy_fn(hypothesis, labels):
    predicted = (hypothesis > 0.5).float()
    accuracy = (predicted == labels).float().mean()
    return accuracy

# %% Training
EPOCHS = 1000
learning_rate = 0.01

W = torch.randn((2, 1), requires_grad=True)
b = torch.randn(1, requires_grad=True)

optimizer = optim.SGD([W, b], lr=learning_rate)

for step in range(EPOCHS):
    for x, labels in dataloader:
        pred = model(x)
        loss = loss_fn(pred, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (step+1) % 100 == 0:
        print('Iter: {:5}, Loss: {:5.4f}'.format(step+1, loss))
        
acc = accuracy_fn(model(xdata), ydata)
print("Accuracy: {:.4f}".format(acc))

