# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 19:48:17 2023

@author: heon
"""

import torch
import numpy as np
import matplotlib.pyplot as plt


# 1
X_data = torch.FloatTensor([-0.4009, 0.5625, 0.5902, -0.1524, -0.5197, 1.8524,
                            1.8365, 2.0741, 0.7872, -1.3902, 0.3354, -1.1721, 0.2123, 1.1060, 1.3120, 0.4242,
-0.7943, 0.3565, -0.5407, -2.1090, 0.8635, -1.4492, 0.3143, -1.0053,
-1.0819, -0.8011, -0.5799, 1.7354, -0.3299, 0.3301, 0.9814, -1.4912,
-1.0249, 0.5230, 0.3451, -0.1033, 0.4315, -0.7729, -0.8030, 0.4645,
-0.3268, -0.0560, 1.2440, 0.9787, 0.7278, 0.0926, -0.1798, 0.7004,
0.7514, -0.2296, 1.7335, -1.1168, 0.7389, -0.4683, -1.5963, -1.6672,
1.4290, 0.2306, -1.8913, 0.7457, 0.8052, 0.7449, -0.1739, 1.1078,
0.0603, 0.6367, 0.0925, -0.6212, 2.0818, 1.0677, -1.4277, -0.3318,
-0.5820, -0.0895, -1.4171, 1.8215, -1.3645, 0.1792, 0.5652, 0.3272,
-0.0472, 0.5974, 0.4486, 1.2600, -0.1637, -1.2390, -1.0557, 1.2461,
-0.7246, -1.0445, 0.5627, 0.1729, 1.9905, -1.1816, 0.3039, 2.2259,
0.3058, -0.6298, 0.7762, -0.5106]).detach()

Y_data = torch.FloatTensor([-1.9025, 0.8607, 0.8564, -1.3496, -1.4961, 6.3983, 3.9409, 3.7766,
1.1055, -3.9978, -0.4221, -4.8150, -0.2143, 3.9791, 0.3136, -0.1594,
-3.4838, -0.3694, -4.5997, -7.3629, 0.8030, -5.4641, 1.8422, -3.5635,
-4.8306, -3.2292, -2.9428, 2.9350, -1.1168, 0.1317, 0.5385, -5.0653,
-2.6164, 0.0374, 0.9324, -2.4492, -1.0155, -3.4148, -5.8199, 0.3838,
-2.6418, -0.3650, 2.6616, 0.4187, 1.3837, -0.9692, -1.3031, 1.8630,
0.7479, -0.9396, 5.2687, -6.9869, 1.9121, -2.7278, -5.1918, -6.6263,
4.9344, -2.0161, -6.5775, 0.9507, 2.4323, 0.5768, -1.9216, 0.5261,
-1.1974, 1.1426, -1.4547, -3.2674, 6.4518, 1.5093, -6.7995, 0.0288,
-3.5536, -1.1744, -4.9784, 5.0087, -7.5694, -0.2519, 0.9493, -0.7502,
-2.5215, 0.5817, 1.2378, 2.9632, -1.9116, -4.3099, -4.4864, 3.6210,
-3.4919, -5.8496, 1.0637, -0.3291, 6.6158, -4.6421, -1.0608, 6.5830,
-0.9454, -2.7748, 1.2265, -3.4573]).detach()

print("Before Training")
W_init = 1.0
b_init = 1.0
print(f'W: {W_init}\t|\tb: {b_init}')

hypothesis = W_init * X_data + b_init

plt.scatter(X_data, Y_data, s=5)
plt.plot(X_data, hypothesis, 'r-')
plt.show()

# 2
cost = torch.mean((Y_data - hypothesis)**2)    

w_list = np.linspace(-3, 9, num=40)
cost_values = []
for idx, w_val in enumerate(w_list):
    hyp = w_val * X_data
    cost_values.append(torch.mean((Y_data - hyp)**2))
    
#plt.plot(w_list, cost_values)
#plt.show()

#LinearRegression -Optimization
learning_rate = 0.001
W = torch.tensor([W_init], requires_grad=True)
b = torch.tensor([b_init], requires_grad=True)

for i in range(5001):
    hypothesis = W * X_data + b
    cost = torch.mean((Y_data - hypothesis)**2)
    cost.backward()
    
    W.data = W.data - learning_rate*W.grad.data
    b.data = b.data - learning_rate*b.grad.data
    
    W.grad.data.zero_()
    b.grad.data.zero_()
    if i % 500 == 0:
        print('epoch: {:5}|\tcost: {:10.4f}|\tW: {:10.2f}|\tb: {:10.2}'.format(i, cost.item(), W.item(), b.item()))
    
print("After Training")
print('W: {:.1f}\t|\tb: {:.1f}'.format(W.item(), b.item()))

plt.scatter(X_data, Y_data, s=5)
plt.plot(X_data, hypothesis.data, 'r-')
plt.show()

print("201821254 심헌")
