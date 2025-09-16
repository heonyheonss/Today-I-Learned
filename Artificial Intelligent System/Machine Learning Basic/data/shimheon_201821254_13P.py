# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 00:10:32 2023

@author: USER
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from matplotlib import pyplot as plt

torch.manual_seed(1)

# %% Training Parameters
EPOCH = 3
BATCH = 100
num_models = 2
LEARNINGRATE = 0.001

# %% MNIST Dataset
data_augmentation = transforms.Compose([
    transforms.RandomRotation(degrees = (-15, 15), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.RandomResizedCrop(size=(28, 28), scale=(0.6, 1.0)),
    transforms.ToTensor()
    ])

train_set = datasets.MNIST(
    root = './data/MNIST',
    train = True,
    transform = data_augmentation, # 데이터를 0에서 255까지 있는 값을 0에서 1사이 값으로 변환
    download = True
)
test_set = datasets.MNIST(
    root = './data/MNIST',
    train = False,
    transform = data_augmentation, # 데이터를 0에서 255까지 있는 값을 0에서 1사이 값으로 변환
    download = True
)

print('number of training data : ', len(train_set))
print('number of test data : ', len(test_set))
print(train_set[0][0].size())

train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=BATCH, shuffle=True)

original_images, labels = next(iter(train_loader))
print("original_images.shape :", original_images.detach().numpy().shape)

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

# %% Plot some sample images
plt.figure(figsize=(10,4))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(train_set[i][0].squeeze().numpy())
    plt.axis('off')
    plt.title(i)

plt.show()

# %% Model define - class
class ConvBNRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride = 1, padding = 1):
        super(ConvBNRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding)
        self.batchnorm = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = F.relu(x)
        return x
    
class DenseBNRelu(nn.Module):
    def __init__(self, in_features, out_features):
        super(DenseBNRelu, self).__init__()
        self.dense = nn.Linear(in_features, out_features)
        self.batchnorm = nn.BatchNorm1d(out_features)
    
    def forward(self, x):
        x = self.dense(x)
        x = self.batchnorm(x)
        x = F.relu(x)
        return x
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBNRelu(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.batchnorm = nn.BatchNorm2d(out_channels)
    
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.batchnorm(x)
        x = x + inputs
        x = F.relu(x)
        return x

# %% class - CNNModel

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = ConvBNRelu(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = ConvBNRelu(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = ConvBNRelu(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool3_flat = nn.Flatten()
        self.dense4 = DenseBNRelu(128* 3 * 3, 256)
        self.drop4 = nn.Dropout(p=0.4)
        self.dense5 = nn.Linear(256, 10)
        
    def forward(self, x):
        net = self.conv1(x)
        net = self.pool1(net)
        net = self.conv2(net)
        net = self.pool2(net)
        net = self.conv3(net)
        net = self.pool3(net)
        net = self.pool3_flat(net)
        net = self.dense4(net)
        net = self.drop4(net)
        net = self.dense5(net)
        return net
# %% class - ResNet

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1_1 = ConvBNRelu(1, 32, kernel_size=3, padding=1)
        self.conv1_2 = ResidualBlock(32, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2_1 = ConvBNRelu(32, 64, kernel_size=3, padding=1)
        self.conv2_2 = ResidualBlock(64, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3_1 = ConvBNRelu(64, 128, kernel_size=3, padding=1)
        self.conv3_2 = ResidualBlock(128, 128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool3_flat = nn.Flatten()
        self.dense4 = DenseBNRelu(128* 3 * 3, 256)
        self.drop4 = nn.Dropout(p=0.4)
        self.dense5 = nn.Linear(256, 10)
        
    def forward(self, x):
        net = self.conv1_1(x)
        net = self.conv1_2(net)
        net = self.pool1(net)
        net = self.conv2_1(net)
        net = self.conv2_2(net)
        net = self.pool2(net)
        net = self.conv3_1(net)
        net = self.conv3_2(net)
        net = self.pool3(net)
        net = self.pool3_flat(net)
        net = self.dense4(net)
        net = self.drop4(net)
        net = self.dense5(net)
        return net

# %% create model
models = []

model = CNNModel()
print(model)
models.append(model)

model = ResNet()
print(model)
models.append(model)

# %% Optimizer
optimizer = optim.Adam(model.parameters(), lr=LEARNINGRATE)
lr_decay = 0.9
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=len(train_set)/BATCH*5, gamma=lr_decay)

# loss
criterion = nn.CrossEntropyLoss()

# Train function
def train(model, images, labels):
    sum_loss = 0
    for model in models:
        pred = model(images)
        loss = criterion(pred, labels)
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        sum_loss += loss
    
    return sum_loss / num_models

# Function to calculate accuracy
def accuracy_fn(models, images, labels):
    logits = models[0](images)
    for i in range(1, num_models):
        logits += models[i](images)
        
    pred = torch.argmax(F.softmax(logits, dim=1), 1)
    correct_prediction = (pred == labels)
    accuracy = correct_prediction.float().mean()
    return accuracy

# %% Training
fstr = "\rEpch: {}, Loss: {:.6f}, Train accuracy: {:.4f}, Test accuracy: {:.4f}"

model.train()
for epoch in range(EPOCH):
    avg_loss = 0.
    avg_train_acc = 0.
    avg_test_acc = 0.
    train_step = 0
    test_step = 0
    for images, labels in train_loader:
        avg_loss += train(models, images, labels)
        print('\r{:6.2f}'.format((train_step)/len(train_loader)*100), end='')
        acc = accuracy_fn(models, images, labels)
        avg_train_acc += acc
        train_step += 1
    avg_loss = avg_loss / train_step
    avg_train_acc = avg_train_acc / train_step
    
    lr_scheduler.step()
    
    with torch.no_grad():
        for images, labels in test_loader:
            acc = accuracy_fn(models, images, labels)
            avg_test_acc += acc
            test_step += 1
            print('\r{:6.2f}'.format((test_step)/len(test_loader)*100), end='')
        avg_test_acc = avg_test_acc / test_step
        
    print(fstr.format(epoch+1, avg_loss, avg_train_acc, avg_test_acc))

print('Learning Finished!')
# %%
print("201821254 심헌")