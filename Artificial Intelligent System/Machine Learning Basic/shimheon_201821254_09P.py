# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 19:58:44 2023

@author: heon
"""

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
EPOCH = 10
BATCH = 128
LEARNINGRATE = 0.001

# %% MNIST Dataset

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

train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=BATCH, shuffle=True)

# %% Plot some sample images
plt.figure(figsize=(10,4))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(train_set[i][0].squeeze().numpy())
    plt.axis('off')
    plt.title(i)

plt.show()

# %% Dfine
# Model
def custom_normal_init(layer):
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_uniform_(layer.weight)
        nn.init.constant_(layer.bias, 0)
        
def Dropout_model():
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 512),
        nn.Dropout(0.2),
        nn.ReLU(),
        nn.Linear(512, 10)
    )
    
    # Apply custom weight initialization
    model.apply(custom_normal_init)
    
    print(model)
    
    return model
    
def without_Dropout_model():
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    )
    
    # Apply custom weight initialization
    model.apply(custom_normal_init)
    
    print(model)
    
    return model

model = Dropout_model()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=LEARNINGRATE)

# Loss
criterion = nn.CrossEntropyLoss()


# Function to calculate accuracy
def accuracy_fn(Y, out):
    pred = torch.argmax(out, 1)
    correct_prediction = (pred == Y)
    accuracy = correct_prediction.float().mean()
    return accuracy

# %% Training
fstr = "Iter: {:5}, Loss: {:5.4f}, Accuracy: {:5.4f}"

for epoch in range(EPOCH):
    for images, labels in train_loader:
        pred = model(images)
        loss = criterion(pred, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    with torch.no_grad():
        train_accuracy = accuracy_fn(labels, F.softmax(model(images), dim=1))
    print(fstr.format(epoch, loss, train_accuracy))
    
# %% Testing
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    
    for data, target in test_loader:
        out = F.softmax(model(data), dim=1)
        preds = torch.argmax(out.data, 1)
        total += len(target)
        correct += (preds==target).sum().item()
        
    print('Test Accuracy: ', correct/total)

print('심헌 201821254')