# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 15:00:04 2023

@author: heon
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader

# %% Input images
image = torch.tensor([[[[ 1.,  2.,  3.,  4.],
                        [ 5.,  6.,  7.,  8.],
                        [ 9., 10., 11., 12.],
                        [13., 14., 15., 16.]]]], dtype=torch.float32)
print("image.shape", image.detach().numpy().shape)
print(image.numpy().reshape(4,4))
plt.imshow(image.numpy().reshape(4,4), cmap = 'gray')
plt.show()

# %% convolution wo padding
print("image.shape", image.detach().numpy().shape)
weight = torch.tensor([[[[1.,1.],
                         [1.,1.]]]], dtype=torch.float32)
print("weight.shape", weight.detach().numpy().shape) # (out_channel, in_channel, kernel_height, kernel_weight)
print(weight.numpy().reshape(2,2))

conv_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, padding="valid", bias=False)
conv_layer.weight.data = weight

conv2d = conv_layer(image)
print("conv2d.shape", conv2d.detach().numpy().shape)
print(conv2d.detach().numpy().reshape(3,3))
plt.imshow(conv2d.detach().numpy().reshape(3,3), cmap="gray")
plt.show()

# %% convolution(zero padding)
print("image.shape", image.detach().numpy().shape)
weight = torch.tensor([[[[1.,1.],
                         [1.,1.]]]], dtype=torch.float32)
print("weight.shape", weight.detach().numpy().shape) # (out_channel, in_channel, kernel_height, kernel_weight)
print(weight.numpy().reshape(2,2))

conv_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, padding="same", bias=False)
conv_layer.weight.data = weight

conv2d = conv_layer(image)
print("conv2d.shape", conv2d.detach().numpy().shape)
print(conv2d.detach().numpy().reshape(4,4))
plt.imshow(conv2d.detach().numpy().reshape(4,4), cmap="gray")
plt.show()

# %% Multi channel Output(zero padding)
print("image.shape", image.detach().numpy().shape)
weight = torch.tensor([[[[1.,1.],
                         [1.,1.]]],
                       [[[10.,10.],
                         [10.,10.]]],
                       [[[-1.,-1.],
                         [-1.,-1.]]],
                       [[[-10.,-10.],
                         [-10.,-10.]]]], dtype=torch.float32)
print("weight.shape", weight.detach().numpy().shape) # (out_channel, in_channel, kernel_height, kernel_weight)

for i in range(4):
    print(weight[i].numpy().reshape(2,2))


conv_layer = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=2, padding="same", bias=False)
conv_layer.weight.data = weight

conv2d = conv_layer(image)
print("conv2d.shape", conv2d.detach().numpy().shape)
feature_maps = np.swapaxes(conv2d, 0, 1)
for i, feature_map in enumerate(feature_maps):
    print(feature_map.view(4,4).detach().numpy())
    plt.subplot(1,4,i+1), plt.imshow(feature_map.view(4,4).detach().numpy(), cmap="gray")
plt.show()
# %% Pooling
print("image.shape", image.detach().numpy().shape)
conv_layer = nn.MaxPool2d(2)

pool2d = conv_layer(image)
print("pool2d.shape", pool2d.detach().numpy().shape)
print(pool2d.detach().numpy())
plt.imshow(pool2d.detach().numpy().reshape(2,2), cmap="gray")
plt.show()
# %% student identification
print("심헌 201821254")