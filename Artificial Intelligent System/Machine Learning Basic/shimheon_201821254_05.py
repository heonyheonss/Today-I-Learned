# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 14:07:06 2023

@author: heon
"""
import numpy as np
import torch
import torch.optim as optim


torch.manual_seed(800) #for reproducibility
# 1. Data generation
data = np.array([
    # x1,  x2,  x3,   y
    [73.,  80.,  75., 152.],
    [93.,  80.,  93., 185.],
    [89.,  91.,  90., 180.],
    [96.,  98., 100., 196.],
    [73.,  66.,  70., 142.],
    ], dtype=np.float32)

X_data = torch.from_numpy(data[:,0:3]) #정보
ground_truth = torch.from_numpy(data[:, [3]]) #목표


# 4. Training
print('--Multivariable linear regression')

W = torch.randn(3, 1, requires_grad=True)
b = torch.randn(1, requires_grad=True)
optimizer = optim.SGD([W, b], lr=0.00004) #0.00004599 #

for epoch in range(500000):
    # Forward pass
    hypothesis = torch.matmul(X_data, W) + b
    cost = torch.mean((ground_truth - hypothesis)**2)
    
    #Backpropagation and optimization
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    if (epoch + 1) % 50000 == 0:
        print('epoch: %d\tcost: %10.6f' % (epoch + 1, cost.item()))

print('W: ', W.data.numpy().flatten())
print('b: ', b.data.numpy())

print('--Predict')
def predict(x):
    return (torch.matmul(x, W) + b).detach().numpy()

x_test = torch.FloatTensor([[70, 60, 50]])
pdV = predict(x_test)
print("result : %10.2f" % pdV[0][0])

print('201821254 심헌')