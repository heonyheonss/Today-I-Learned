# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 15:41:42 2023

@author: heon
"""

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

# %% FashionMNIST Dataset

train_set = datasets.FashionMNIST(
    root = './data/FashionMNIST',
    train = True,
    transform = transforms.ToTensor(), # 데이터를 0에서 255까지 있는 값을 0에서 1사이 값으로 변환
    download = True
)
test_set = datasets.FashionMNIST(
    root = './data/FashionMNIST',
    train = False,
    transform = transforms.ToTensor(), # 데이터를 0에서 255까지 있는 값을 0에서 1사이 값으로 변환
    download = True
)

print('number of training data : ', len(train_set))
print('number of test data : ', len(test_set))
print(train_set[0][0].size())

train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=BATCH, shuffle=True)

# %% Dfine
# Model
def custom_normal_init(layer):
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_uniform_(layer.weight)
        nn.init.constant_(layer.bias, 0)
        
def create_model():
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
    
model = create_model()

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
    
# %% Plot some sample images
#%%
match_list = ["T-shirt","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle Boot"]

#%% custom imshow
def custom_imshow(img, i):
    plt.subplot(2, 5, i+1)
    img = img.numpy()
    plt.imshow(np.transpose(img, (1, 2, 0)))
    
def process():
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        custom_imshow(inputs[0])


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
    
    title_fstr = "{}\n{}"

    plt.figure(figsize=(10,4))
    plt.subplots(constrained_layout=True)
    for i in range(10):
        custom_imshow(data[i], i)
        plt.axis('off')
        plt.title(title_fstr.format(match_list[preds[i].item()], match_list[target[i].item()]))
        
    plt.show()
    

    print('Test Accuracy: ', correct/total)
    

print('심헌 201821254')