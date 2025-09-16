# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 00:31:52 2023

@author: USER
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

x_data = torch.FloatTensor([1, 2, 3, 4, 5])
y_data = torch.FloatTensor([1, 2, 3, 4, 5])
print(x_data)

print("--Hypothesis")
W_init = 3.0
b_init = 0.5
print('Parameters value: ', W_init, b_init)

hypothesis = W_init * x_data + b_init
print('Hypothesis value: ', hypothesis)

plt.plot(x_data, hypothesis, 'r-')
plt.plot(x_data, y_data, 'o')
plt.ylim(0, 8)
plt.show()

W = torch.tensor([W_init], requires_grad=True)
b = torch.tensor([b_init], requires_grad=True)

print("--Cost")
print('torch.mean((y - predict)**2): x_data=', x_data, '->', \
      torch.mean((y_data - W*x_data -b)**2).item())

hypothesis = W * x_data + b
cost = torch.mean((y_data - hypothesis)**2)    
cost.backward()

print(f'W:{W.item()},b: {b.item()}')
print(f'W_grad: {W.grad.data}, b_grad: {b.grad.data}')
print('cost =', cost.item())

w_list = np.linspace(-3, 5, num=15)
cost_values = []
for idx, w_val in enumerate(w_list):
    hyp = w_val * x_data
    cost_values.append(torch.mean((y_data - hyp)**2))
    
plt.plot(w_list, cost_values)
plt.xlabel('W')
plt.ylabel('Cost')
plt.show()

# Linear Regression - Optimization (one step)
print('--One step optimization')
learning_rate = 0.01
W = torch.tensor([W_init], requires_grad=True)
b = torch.tensor([b_init], requires_grad=True)

hypothesis = W * x_data + b
cost = torch.mean((y_data - hypothesis)**2)
cost.backward()

print('--Update parameters')
print('Previous parameters: ', W.item(), b.item())
print('gradient: ', W.item(), b.item())
W.data = W.data - learning_rate*W.grad.data
b.data = b.data - learning_rate*b.grad.data
print('Updated parameters: {:.1f}{:.1f}'.format(W.item(), b.item()))

#LinearRegression -Optimization
print('--Simple linear regression')
learning_rate = 0.01

W = torch.tensor([W_init], requires_grad=True)
b = torch.tensor([b_init], requires_grad=True)

for i in range(100):
    hypothesis = W * x_data + b
    cost = torch.mean((y_data - hypothesis)**2)
    cost.backward()
    W.data = W.data - learning_rate*W.grad.data
    b.data = b.data - learning_rate*b.grad.data
    
    W.grad.data.zero_()
    b.grad.data.zero_()
    if i % 10 == 0:
        print('{:5}|{:10.4f}|{:10.4f}|{:10.6}'.format(i, W.item(), b.item(), cost.item()))
    

plt.plot(x_data, y_data, 'o')
plt.plot(x_data, hypothesis.data, 'r-')
plt.ylim(0,8)
plt.show()

# Predict
print('--Predict')
print(5, '->', (W * 5 + b).item())
print(2.5, '->', (W * 2.5 + b).item())


print('201821254 심헌')