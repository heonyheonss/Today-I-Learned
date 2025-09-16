# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 22:25:45 2023

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

torch.manual_seed(3)

# %%
image = torch.tensor([[[[1.,2.,3.],
                       [4.,5.,6.],
                       [7.,8.,9.]]]], dtype=torch.float32)

# Display the input image
print("image.shape", image.detach().numpy().shape) #(batch size, channel number, height, weight)
plt.imshow(image.numpy().reshape(3,3), cmap = 'gray')
plt.show()

# %% 1st conv
print("image.shape", image.detach().numpy().shape)
weight = torch.tensor([[[[1.,1.],
                         [1.,1.]]]], dtype=torch.float32)
print("weight.shape", weight.detach().numpy().shape) # (out_channel, in_channel, kernel_height, kernel_weight)

conv_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, padding="valid", bias=False)
conv_layer.weight.data = weight

conv2d = conv_layer(image)
print("conv2d.shape", conv2d.detach().numpy().shape)
print(conv2d.detach().numpy().reshape(2,2))
plt.imshow(conv2d.detach().numpy().reshape(2,2), cmap="gray")
plt.show()
# %% zero padding
print("image.shape", image.detach().numpy().shape)
weight = torch.tensor([[[[1.,1.],
                         [1.,1.]]]], dtype=torch.float32)
print("weight.shape", weight.detach().numpy().shape) # (out_channel, in_channel, kernel_height, kernel_weight)

conv_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, padding="same", bias=False)
conv_layer.weight.data = weight

conv2d = conv_layer(image)
print("conv2d.shape", conv2d.detach().numpy().shape)
print(conv2d.detach().numpy().reshape(3,3))
plt.imshow(conv2d.detach().numpy().reshape(3,3), cmap="gray")
plt.show()
# %% Multi channel Output
print("image.shape", image.detach().numpy().shape)
weight = torch.tensor([[[[1.,1.],
                         [1.,1.]]],
                       [[[10.,10.],
                         [10.,10.]]],
                       [[[-1.,-1.],
                         [-1.,-1.]]]], dtype=torch.float32)
print("weight.shape", weight.detach().numpy().shape) # (out_channel, in_channel, kernel_height, kernel_weight)

conv_layer = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=2, padding="same", bias=False)
conv_layer.weight.data = weight

conv2d = conv_layer(image)
print("conv2d.shape", conv2d.detach().numpy().shape)
feature_maps = np.swapaxes(conv2d, 0, 1)
for i, feature_map in enumerate(feature_maps):
    print(feature_map.view(3,3).detach().numpy())
    plt.subplot(1,3,i+1), plt.imshow(feature_map.view(3,3).detach().numpy(), cmap="gray")
plt.show()
# %% Pooling
image = torch.tensor([[[[4.,3.],
                        [2.,1.]]]], dtype=torch.float32)
print("image.shape", image.detach().numpy().shape)
conv_layer = nn.MaxPool2d(2)

pool2d = conv_layer(image)
print("pool2d.shape", pool2d.detach().numpy().shape)
print(pool2d.detach().numpy())
# %% MNIST Example

train_set = datasets.MNIST(
    root = './data/MNIST',
    train = True,
    transform = transforms.ToTensor(), # 데이터를 0에서 255까지 있는 값을 0에서 1사이 값으로 변환
    download = True
)
test_set = datasets.MNIST(
    root = './data/MNIST',
    train = False,
    transform = transforms.ToTensor(), # 데이터를 0에서 255까지 있는 값을 0에서 1사이 값으로 변환
    download = True
)

print('number of training data : ', len(train_set))
print('number of test data : ', len(test_set))
print(train_set[0][0].size())

train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=1, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=1, shuffle=True)

class_names = ['0', '1', '2', '3', '4', '5', '6 ', '7', '8', '9']

for img, label in train_loader:
    img = img[0] # Take the first image in the batch
    label = label[0]
    break

# Display the first img
plt.imshow(img[0], cmap="gray")
plt.title(label.item())
plt.show()

# Create the Conv2D Layer in Pytorch

conv_layer = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, stride=2, padding=1, bias=False)
weight_init = torch.nn.init.normal_(torch.empty(5, 1, 3, 3), mean=0, std=0.01)
conv_layer.weight.data = weight_init

img = img.float().unsqueeze(0) # Add batch dimension
print("img.shape", image.detach().numpy().shape)
conv2d = conv_layer(img)
print("conv2d.shape", conv2d.detach().numpy().shape)

feature_maps = conv2d[0].detach().numpy()
for i, feature_map in enumerate(feature_maps):
    plt.subplot(1,5,i+1), plt.imshow(feature_map, cmap="gray")
plt.show()

# Apply MaxPool2D
max_pool = F.max_pool2d(conv2d, kernel_size=2, stride=2, padding=0)
print("max_pool.shape", max_pool.detach().numpy().shape)

# Display the pooled feature maps
feature_maps = max_pool[0].detach().numpy()
for i, feature_map in enumerate(feature_maps):
    plt.subplot(1,5,i+1), plt.imshow(feature_map, cmap="gray")
plt.show()
#%%
print("201821254 심헌")
