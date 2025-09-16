# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 00:44:20 2023

@author: USER
"""

import os


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)

yx = np.loadtxt('data/wine/wine.data', delimiter=',', dtype = np.float32)
np.random.shuffle(yx)

print(yx.shape)
print(yx[:10, :])

n_data = len(yx)

x_train = torch.tensor(yx[:-int(n_data/5), 1:])
y_train = torch.tensor(yx[:-int(n_data/5), 0])

print(y_train)

# %%
x_test = torch.tensor(yx[-int(n_data/5):, 1:])
y_test = torch.tensor(yx[-int(n_data/5):, 0])

print(torch.unique(y_train).numpy())

# Convert labels to one-hot encoding using Pytorch
nb_classes = 3

y_train = F.one_hot(y_train.to(torch.int64) - 1, num_classes=nb_classes).float()
y_test = F.one_hot(y_test.to(torch.int64) - 1, num_classes=nb_classes).float()

print(y_train[:5, :])

print('x_train:', x_train.shape, '\ty_train:', y_train.shape)
print('x_test:', x_test.shape, '\ty_test:', y_test.shape)

# Define normalization function
def normalization(d_train, d_test=None):
    min_vars = torch.min(d_train, dim=0).values
    max_vars = torch.max(d_train, dim=0).values
    
    if d_test is None:
        return (d_train-min_vars)/(max_vars-min_vars)
    else:
        return (d_train-min_vars)/(max_vars-min_vars), (d_test-min_vars)/(max_vars-min_vars)
    

# Normalize the data
x_train_nrm, x_test_nrm = normalization(x_train, x_test)

# Define standardization function
def standardization(d_train, d_test=None):
    mean_vars = torch.mean(d_train, dim=0)
    std_vars = torch.std(d_train, dim=0)
    
    if d_test is None:
        return (d_train-mean_vars)/std_vars
    else:
        return (d_train-mean_vars)/std_vars, (d_test-mean_vars)/std_vars
    
# Standardization the data
x_train_std, x_test_std = standardization(x_train, x_test)
print('mean(x_std): %.4f\t std(x_Std): %.4f' % (torch.mean(x_train_std), torch.std(x_train_std)))

# Plotting
plt.figure(figsize=(20,10))
for i in range(x_train.shape[1]):
    plt.subplot(3,5,i+1)
    plt.plot(x_train[:,i])
    plt.title('original x%d' %(i+1))
plt.show()

plt.figure(figsize=(20,10))
for i in range(x_train_nrm.shape[1]):
    plt.subplot(3,5,i+1)
    plt.plot(x_train_nrm[:,i])
    plt.title('normalized x%d' %(i+1))
plt.show()

plt.figure(figsize=(20,10))
for i in range(x_train_std.shape[1]):
    plt.subplot(3,5,i+1)
    plt.plot(x_train_std[:,i])
    plt.title('standard x%d' %(i+1))
plt.show()


# Dataload for training
dataset = TensorDataset(x_train_std, y_train)
dataloader = DataLoader(dataset, batch_size=len(x_train_std))

# Define
def model(x):
    return F.softmax(torch.matmul(x, W) + b, dim=1)

def loss_fn(pred, Y):
    cost = F.cross_entropy(pred, Y)
    return cost

def accuracy_fn(hypothesis, labels):
    accuracy = (hypothesis == labels).float().mean()
    return accuracy

# Training
EPOCHS = 1000

W = torch.randn(x_train.shape[1], nb_classes, dtype=torch.float32, requires_grad=True)
b = torch.zeros(nb_classes, dtype=torch.float32, requires_grad=True)
variables = [W, b]

print(W.shape)
print(b.shape)

lr_init = 1.
lr_decay = 0.99
optimizer = optim.SGD(variables, lr = lr_init)

lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)

for step in range(EPOCHS):
    
    
    for x, labels in dataloader:
        pred = model(x)
        loss = loss_fn(pred, torch.argmax(labels,1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (step+1) % 100 == 0:
        pred_train = torch.argmax(model(x_train_std), dim=1)
        Y_train = torch.argmax(y_train, dim=1)
        train_acc = accuracy_fn(pred_train, Y_train)
        pred_test = torch.argmax(model(x_test_std), dim=1)
        Y_test = torch.argmax(y_test, dim=1)
        test_acc = accuracy_fn(pred_test, Y_test)
        
        print(optimizer.param_groups[0]['lr'])
        print('Epoch:%d \tloss: %0.4f \ttrain acc:%0.4f \ttest acc:%0.4f'%(step+1, loss, train_acc, test_acc))
    
    lr_scheduler.step()
    
print("201821254 심헌")