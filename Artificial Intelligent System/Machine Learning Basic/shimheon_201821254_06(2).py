# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(0)

xy = np.loadtxt('data/data-04-zoo.csv', delimiter=',',dtype=np.float32)

x_train = xy[:, 0:16]
y_train = xy[:, [16]]
x_data = torch.tensor(x_train, dtype=torch.float32)
y_data = torch.tensor(y_train, dtype=torch.float32)

print(y_data.shape)

nb_classes = 7

y_data = F.one_hot(y_data.squeeze().to(torch.int64), num_classes=nb_classes)
print(y_data.shape)

# Define
def model(x):
    #Softmax Regression
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

W = torch.zeros(16, nb_classes, dtype=torch.float32, requires_grad=True)
b = torch.zeros(nb_classes, dtype=torch.float32, requires_grad=True)
variables = [W, b]
optimizer = optim.SGD(variables, lr=learning_rate)

for step in range(EPOCHS):
    loss = loss_fn(x_data, y_data)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (step+1) % 100 == 0:
        print('Epoch: {:5}, Loss: {:5.4f}'.format(step+1, loss))

print('Accuracy: ', accuracy_fn(torch.argmax(model(x_data), dim=1), torch.argmax(y_data, dim=1)).item())
print("201821254 심헌")
