# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 23:51:27 2023

@author: USER
"""

import torch
import torch.optim as optim
import matplotlib.pyplot as plt

torch.manual_seed(42) # random seed

# 1. Generate data
x_train = [[1., 2.],
           [2., 3.],
           [3., 1.],
           [4., 3.],
           [5., 3.],
           [6., 2.]]
y_train = [[0.],
           [0.],
           [0.],
           [1.],
           [1.],
           [1.]]

x_test = [[5., 2.]]
y_test = [[1.]]

x1 = [x[0] for x in x_train]
x2 = [x[1] for x in x_train]

colors = [int(y[0] % 3) for y in y_train]
plt.scatter(x1, x2, c=colors, marker='^')
plt.scatter(x_test[0][0], x_test[0][1], c='red')

plt.xlabel('x1')
plt.xlabel('x2')
plt.show()

# Create a Dataloader for batching
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

from torch.utils.data import TensorDataset, DataLoader
dataset = TensorDataset(x_train, y_train)
dataloader = DataLoader(dataset, batch_size=len(x_train), shuffle=True)

# Define
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

# Training
EPOCHS = 1000
learning_rate = 0.01

W = torch.zeros((2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

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

# validation
x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
test_acc = accuracy_fn(model(x_test), y_test)
print("Testset Accuracy: {:.4f}".format(test_acc.item()))
print("TEst result: {:.3f}".format(model(x_test)[0,0].item()))

print("201821254 심헌")







