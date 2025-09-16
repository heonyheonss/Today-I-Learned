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
    hypothesis = 1 / (1 + torch.exp(torch.matmul(x, W) + b))
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

# %% plot - pytorch
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create a grid of points
x1 = torch.arange(0, 1, 0.01)
x2 = torch.arange(0, 1, 0.01)
X1, X2 = torch.meshgrid(x1, x2)
grid = torch.cat((X1.reshape(-1, 1), X2.reshape(-1, 1)), dim=1).float()

# Use the trained model to make predictions on the grid
with torch.no_grad():
    h = model(grid)
H = h.view(100, 100).numpy()

# Create a 3D surface plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
surf = ax.plot_surface(X1.numpy(), X2.numpy(), H, cmap='coolwarm', linewidth=0, antialiased=False)

ax.set_zlim(-0.2, 1.1)
fig.colorbar(surf, shrink=0.5, aspect=5)
fig.tight_layout()

plt.show()

# %% Hard way to make MLP
W1 = torch.randn(2, 2, requires_grad=True)
b1 = torch.randn(2, requires_grad=True)
W2 = torch.randn(2, 2, requires_grad=True)
b2 = torch.randn(1, requires_grad=True)

trainable_vars = [W1, b1, W2, b2]

# Define the model
def model(x):
    l1 = torch.sigmoid(torch.matmul(x, W1) + b1)
    hypothesis = torch.sigmoid(torch.matmul(l1, W2) + b2)
    return hypothesis

# %% Use pytorch to make a MLP
def create_model():
    model = nn.Sequential(
        nn.Linear(2, 2),
        nn.Sigmoid(),
        nn.Linear(2,1)
    )
    
    print(model)
    return model

model = create_model()

#%% Training
EPOCHS = 1000
learning_rate = 0.1
criterion = nn.BCEWithLogitsLoss()
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
        
acc = accuracy_fn(torch.sigmoid(model(xdata)), ydata)
print("Accuracy: {:.4f}".format(acc))

# %% plot - pytorch
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create a grid of points
x1 = torch.arange(0, 1, 0.01)
x2 = torch.arange(0, 1, 0.01)
X1, X2 = torch.meshgrid(x1, x2)
grid = torch.cat((X1.reshape(-1, 1), X2.reshape(-1, 1)), dim=1).float()

# Use the trained model to make predictions on the grid
with torch.no_grad():
    h = torch.sigmoid_(model(grid))
H = h.view(100, 100).numpy()

# Create a 3D surface plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
surf = ax.plot_surface(X1.numpy(), X2.numpy(), H, cmap='coolwarm', linewidth=0, antialiased=False)

ax.set_zlim(-0.2, 1.1)
fig.colorbar(surf, shrink=0.5, aspect=5)
fig.tight_layout()

plt.show()

print('심헌 201821254')


