# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 00:22:16 2023

@author: USER
"""

import torch
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(42) # random seed

# 1. Generate data
x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 7, 7]]

y_data = [2,
          2,
          2,
          1,
          1,
          1,
          0,
          0]

# convert into tensor and float format
x_data = torch.tensor(x_data, dtype=torch.float32)
y_data = torch.tensor(y_data, dtype=torch.float32)

print(torch.unique(y_data))

nb_classes = 3

print(y_data)
y_data = F.one_hot(y_data.to(torch.int64), num_classes=nb_classes)
print(y_data)

print('x shape: ', x_data.shape)
print('y shape: ', y_data.shape)

# Define
def model(x):
    #Logistic Regression
    return F.softmax(torch.matmul(x, W) + b, dim=1)

def loss_fn(x, y):
    logits = model(x)
    cost = -torch.sum(y * torch.log(logits), dim=1)
    cost_mean = torch.mean(cost)
    return cost_mean

def accuracy_fn(predicted, labels):
    correct_predictions = (predicted == labels).float()
    accuracy = torch.mean(correct_predictions)
    return accuracy

# Training
EPOCHS = 2000

learning_rate = 0.1

W = torch.randn(4, nb_classes, dtype=torch.float32, requires_grad=True)
b = torch.zeros(nb_classes, dtype=torch.float32, requires_grad=True)
variables = [W, b]
optimizer = optim.SGD(variables, lr=learning_rate)

for step in range(EPOCHS):
    loss = loss_fn(x_data, y_data)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (step+1) % 100 == 0:
        print('Iter: {:5}, Loss: {:5.4f}'.format(step+1, loss))

print('Label: ', torch.argmax(y_data, dim=1))
print('Pred: ', torch.argmax(model(x_data), dim=1))
print('Accuracy: ', accuracy_fn(torch.argmax(model(x_data), dim=1), torch.argmax(y_data, dim=1)).item())
print("201821254 심헌")
