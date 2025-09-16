# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 02:19:27 2023

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

# %%
traindata = np.loadtxt('data/data09/traindata.csv', delimiter=',', dtype=np.float32)
testdata = np.loadtxt('data/data09/testdata.csv', delimiter=',', dtype=np.float32)

print(traindata)

x_train = torch.tensor(traindata[:, [0]])
y_train = torch.tensor(traindata[:, [-1]])
x_test = torch.tensor(testdata[:, [0]])
y_test = torch.tensor(testdata[:, [-1]])

plt.plot(x_train, y_train, 'b.')
plt.title('train')
plt.show()
plt.plot(x_test, y_test, 'r.')
plt.title('test')
plt.show()

# %% Model define

# Random normal initializer
def custom_normal_init(layer):
    if isinstance(layer, nn.Linear):
        nn.init.normal_(layer.weight, mean=0, std=1)
        nn.init.constant_(layer.bias, 0)
        
def create_model():
    model = nn.Sequential(
        nn.Linear(in_features=x_train.shape[1], out_features=300),
        nn.Sigmoid(),
        nn.Linear(in_features=300, out_features=300),
        nn.Sigmoid(),
        nn.Linear(in_features=300, out_features=300),
        nn.Sigmoid(),
        nn.Linear(in_features=300, out_features=1)
    )
    
    # Apply custom weight initialization
    model.apply(custom_normal_init)
    
    print(model)
    return model

model = create_model()


# %% Define
dataset = TensorDataset(x_train, y_train)
dataloader = DataLoader(dataset, batch_size=len(x_train), shuffle=True)

def loss_fn(hypothesis, labels, model):
    mse_loss = F.mse_loss(hypothesis, labels)
    l2_loss = 0.1 * (torch.norm(model[0].weight, p=2) +
                     torch.norm(model[2].weight, p=2) +
                     torch.norm(model[4].weight, p=2) +
                     torch.norm(model[6].weight, p=2))
    cost = mse_loss + l2_loss
    return cost

def loss_wo_reg_fn(hypothesis, labels):
    return F.mse_loss(hypothesis, labels)

# %% Training without regularization
EPOCHS = 5000
lr_init = 0.02
lr_decay = 0.99

optimizer = optim.Adam(model.parameters(), lr=lr_init)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=lr_decay)

fstr = "Iter: {:5}, loss_train: {:1.4f}, loss_test: {:1.4f}, lr: {:1.4f}"

for steps in range(EPOCHS):
    for x, labels in dataloader:
        pred = model(x_train)
        optimizer.zero_grad()
        loss = loss_wo_reg_fn(pred, y_train)
        loss.backward()
        optimizer.step()
        
    if (steps + 1) % 500 == 0:
        train_loss = loss_wo_reg_fn(model(x_train), y_train)
        test_loss = loss_wo_reg_fn(model(x_test), y_test)
        current_lr = optimizer.param_groups[0]['lr']
        print(fstr.format(steps + 1, train_loss, test_loss, current_lr))
        
    lr_scheduler.step()

# Plot results without regularization
plt.figure()
plt.plot(x_train, y_train, 'b.')
plt.plot(x_train, model(x_train).detach().numpy(), 'bx-')
plt.title('train')
plt.grid()
plt.figure()
plt.plot(x_test, y_test, 'r.')
plt.plot(x_test, model(x_test).detach().numpy(), 'rx-')
plt.title('test')
plt.legend(['label', 'pred'])
plt.grid()
plt.show()

# %% Training with regularization
model = create_model()

EPOCHS = 10000

lr_init = 0.02
lr_decay = 0.98

optimizer = optim.Adam(model.parameters(), lr=lr_init)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=lr_decay)

fstr = "Iter: {:5}, loss_train: {:1.4f}, loss_test: {:1.4f}, lr: {:1.4f}"

for steps in range(EPOCHS):
    for x, labels in dataloader:
        pred = model(x_train)
        optimizer.zero_grad()
        loss = loss_fn(pred, y_train, model)
        loss.backward()
        optimizer.step()
        
    if (steps + 1) % 500 == 0:
        train_loss = loss_fn(model(x_train), y_train, model)
        test_loss = loss_fn(model(x_test), y_test, model)
        current_lr = optimizer.param_groups[0]['lr']
        print(fstr.format(steps + 1, train_loss, test_loss, current_lr))
        
    lr_scheduler.step()

# Plot results without regularization
plt.figure()
plt.plot(x_train, y_train, 'b.')
plt.plot(x_train, model(x_train).detach().numpy(), 'bx-')
plt.title('train')
plt.grid()
plt.figure()
plt.plot(x_test, y_test, 'r.')
plt.plot(x_test, model(x_test).detach().numpy(), 'rx-')
plt.title('test')
plt.legend(['label', 'pred'])
plt.grid()
plt.show()
print("201821254 심헌")