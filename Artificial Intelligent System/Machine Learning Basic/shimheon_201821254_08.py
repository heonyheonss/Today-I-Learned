# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 15:03:26 2023

@author: heon
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

#X = torch.FloatTensor(np.loadtxt('data/auto_mpg.csv', delimiter=',', dtype = np.float32))
X = np.loadtxt('data/auto_mpg.csv', delimiter=',', dtype = np.float32)

mpg = torch.FloatTensor(X[:,[0]])
data = torch.FloatTensor(X[:,1:])


# %% standardization
def standardization(d_train, d_test=None):
    mean_vars = torch.mean(d_train, dim=0)
    std_vars = torch.std(d_train, dim=0)
    
    if d_test is None:
        return (d_train-mean_vars)/std_vars
    else:
        return (d_train-mean_vars)/std_vars, (d_test-mean_vars)/std_vars
    
# Define normalization function
def normalization(d_train, d_test=None):
    min_vars = torch.min(d_train, dim=0).values
    max_vars = torch.max(d_train, dim=0).values
    
    if d_test is None:
        return (d_train-min_vars)/(max_vars-min_vars)
    else:
        return (d_train-min_vars)/(max_vars-min_vars), (d_test-min_vars)/(max_vars-min_vars)

    
# Standardization the data
data_std = standardization(data)
mpg_std = standardization(mpg)
# X_std = standardization(X)

# Normalization the data
data_norm = normalization(data)
mpg_norm = normalization(mpg)

dataset = TensorDataset(data_norm, mpg_norm)
dataloader = DataLoader(dataset, batch_size=len(data_std), shuffle=True)


# %% definitions

# Random normal initializer
def custom_normal_init(layer):
    if isinstance(layer, nn.Linear):
        nn.init.normal_(layer.weight, mean=0, std=1)
        nn.init.constant_(layer.bias, 0)

# linear model
def linear_model():
    model = nn.Sequential(
        nn.Linear(in_features=6, out_features=1)
        )
    
    print(model)
    return model

# MLP model        
def create_model():
    model = nn.Sequential(
        nn.Linear(in_features=data.shape[1], out_features=300),
        nn.Sigmoid(),
        nn.Linear(in_features=300, out_features=1)
    )
    
    # Apply custom weight initialization
    model.apply(custom_normal_init)
    
    print(model)
    return model

def loss_fn(hypothesis, labels):
    return F.mse_loss(hypothesis, labels)

#%% MLP Training
model = create_model()

EPOCHS = 5100
learning_rate = 0.01
criterion = nn.MSELoss()
# criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for step in range(EPOCHS):
    for x, labels in dataloader:
        logit = model(x)
        # loss = loss_fn(torch.sigmoid(logit), labels)
        loss = criterion(logit, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (step+1) % 100 == 0:
        print('Iter: {:5}, Loss: {:5.4f}'.format(step+1, loss))


# %% Linear model
model_linear = linear_model()

EPOCHS = 1000
learning_rate = 0.1
criterion = nn.MSELoss()
# criterion = nn.BCELoss()
optimizer = optim.Adam(model_linear.parameters(), lr=learning_rate)

for step in range(EPOCHS):
    for x, labels in dataloader:
        logit = model_linear(x)
        # loss = loss_fn(torch.sigmoid(logit), labels)
        loss = criterion(logit, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (step+1) % 100 == 0:
        print('Iter: {:5}, Loss: {:5.4f}'.format(step+1, loss))
        