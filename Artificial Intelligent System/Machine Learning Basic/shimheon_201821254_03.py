# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 14:54:14 2023

@author: heon
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

#1
mat44_A = torch.FloatTensor([[16, 15, 14, 13], [12, 11, 10, 9], [8, 7, 6, 5], [4, 3, 2, 1]])
print(mat44_A.numpy())
mat13_b1 = torch.FloatTensor(mat44_A[1, 1:])
mat13_b2 = torch.FloatTensor(mat44_A[3, 1:])
mat23_B = np.array([mat13_b1.numpy(), mat13_b2.numpy()])
mat23_B = torch.FloatTensor(mat23_B)
x = mat44_A[[1,3],1:]
print(x.numpy())
print(mat23_B.numpy())


#2
p = mat23_B.sum(dim=0)
print(p.numpy())
plt.plot(p)
plt.show()

#3
ft = torch.FloatTensor([[[0, 1, 2], [3, 4, 5]],
                        [[6, 7, 8], [9, 10, 11]],
                        [[12, 13, 14], [15, 16, 17]]
                        ])
print(ft.view(-1, 3).numpy())
print(ft.view(-1, 3).shape)
print(ft.view(-1, 1, 3).numpy())
print(ft.view(-1, 1, 3).shape)