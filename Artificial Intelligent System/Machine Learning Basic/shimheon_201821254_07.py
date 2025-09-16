# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 15:03:19 2023

@author: heon
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

torch.manual_seed(42)

iris = load_iris()

X_data = torch.tensor(iris['data'], dtype=torch.float32)
print('X_data')
print(iris['feature_names'])
print(X_data[:5])

Y_data = torch.tensor(iris['target'], dtype=torch.float32)
print('Y_data')
print(iris['target_names'])
print(Y_data[:5])

nb_classes = 3

y_train = F.one_hot(Y_data.to(torch.int64), num_classes=nb_classes).float()
#%%
print(y_train)
# %%
# Define standardization function
def standardization(d_train, d_test=None):
    mean_vars = torch.mean(d_train, dim=0)
    std_vars = torch.std(d_train, dim=0)
    
    if d_test is None:
        return (d_train-mean_vars)/std_vars
    else:
        return (d_train-mean_vars)/std_vars, (d_test-mean_vars)/std_vars
    
# Standardization the data
X_data_std = standardization(X_data)
# print('mean(x_std): %.4f\t std(x_Std): %.4f' % (torch.mean(x_train_std), torch.std(x_train_std)))

# %%Dataload for training
dataset = TensorDataset(X_data_std, y_train)
dataloader = DataLoader(dataset, batch_size=len(X_data_std))

# Define
def model(x):
    return torch.matmul(x, W) + b

def loss_fn(pred, Y):
    cost = F.cross_entropy(pred, Y)
    return cost

def accuracy_fn(hypothesis, labels):
    accuracy = (hypothesis == labels).float().mean()
    return accuracy

# Training
EPOCHS = 1000

W = torch.randn(X_data.shape[1], nb_classes, dtype=torch.float32, requires_grad=True)
b = torch.zeros(nb_classes, dtype=torch.float32, requires_grad=True)
variables = [W, b]

print(W.shape)
print(b.shape)

lr_init = 1.
lr_decay = 0.99
optimizer = optim.SGD(variables, lr = lr_init)

lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)

# %%
for step in range(EPOCHS):
    
    
    for x, labels in dataloader:
        pred = model(x)
        loss = loss_fn(pred, torch.argmax(labels,1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (step+1) % 100 == 0:
        pred_train = torch.argmax(model(X_data_std), dim=1)
        Y_train = torch.argmax(y_train, dim=1)
        train_acc = accuracy_fn(pred_train, Y_train)
        
        #print(optimizer.param_groups[0]['lr'])
        print('epoch:%d \tcost: %0.4f \taccuracy:%0.4f \tlr:%0.4f'%(step+1, loss, train_acc, optimizer.param_groups[0]['lr']))
    
    lr_scheduler.step()
    
print("201821254 심헌")