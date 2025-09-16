# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 20:04:40 2023

@author: USER
"""

# Multi-variable Linear Regression
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42) #for reproducibility

# 1. Data generation
print("--Data Generation")
W_gt = [1., 2., 3.]
b_gt = 1.

X_data = torch.rand((1000, 3))
Y_data = W_gt[0]*X_data[:, [0]]+W_gt[1]*X_data[:, [1]]+W_gt[2]*X_data[:, [2]] + b_gt + torch.normal(0., 1.,(1000,1))
# torch.normal(mean, std, size); 정규 분포에서 무작위 샘플 추출

ground_truth = W_gt[0]*X_data[:,[0]]+W_gt[1]*X_data[:,[1]]+W_gt[2]*X_data[:,[2]] + b_gt
# H(X) = w1*x1 + w2*x2 + w3*x3 + b

print(X_data.shape, Y_data.shape, ground_truth.shape)


# 2. Model
print('--Hypothesis')
W = torch.randn(3,1,requires_grad=True)
b = torch.randn(1,requires_grad=True)
print('W: ', W.data.numpy())
print('b: ', b.data.numpy())

hypothesis = torch.matmul(X_data, W) + b


# 3. Optimizer % Cost
print('--Optimizer % Cost')
# Define the learning rate and create an optimizer
learning_rate = 0.1

# Update 1 epoch
W = torch.randn(3,1,requires_grad=True)
b = torch.randn(1,requires_grad=True)

optimizer = optim.SGD([W, b], lr=learning_rate)

print('--Update Parameters')
print('Previous Parameters')
print('W: ', W.data.numpy().flatten())
print('b: ', b.data.numpy())

# Forward pass
hypothesis = torch.matmul(X_data, W) + b
# Define the cost function (Mean Square Error)
cost = torch.mean((Y_data - hypothesis)**2)
# cost = torch.mean((ground_truth - hypothesis)**2)

#Backpropagation and optimization
optimizer.zero_grad() # gradient 초기화
cost.backward() # gradient 계산
optimizer.step() # step으로 계산
# W.data = W.data-lr*@.grad.data

print('Updated parameters')
print('W: ', W.data.numpy().flatten())
print('b: ', b.data.numpy())


# 4. Training
print('--Multivariable linear regression')

W = torch.randn(3,1,requires_grad=True)
b = torch.randn(1,requires_grad=True)
optimizer = optim.SGD([W, b], lr=0.01)

for i in range(10000):
    # Forward pass
    hypothesis = torch.matmul(X_data, W) + b
    cost = torch.mean((Y_data - hypothesis)**2)
    # cost = torch.mean((ground_truth - hypothesis)**2)
    
    #Backpropagation and optimization
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    if (i + 1) % 10 == 0:
        print('epoch: %d\tcost: %10.6f' % (i + 1, cost.item()))

print('W: ', W.data.numpy().flatten())
print('b: ', b.data.numpy())

# 5. Validation
# Prediction
print('--Predict')
def predict(x):
    return (torch.matmul(x, W) + b).detach().numpy()

print('Prediction results\n', predict(X_data)[:10, :])
print('Ground truth\n', ground_truth[:10, :])
print('Y_data\n', Y_data[:10, :])

new_data = torch.tensor([[1., 1., 1.], [1., 2., 3.]], dtype=torch.float32)
print('--New data prediction')
print(predict(new_data))

print('201821254 심헌')