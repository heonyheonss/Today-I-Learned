# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 15:02:30 2023

@author: heon
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader

# %% Training Parameter
EPOCH = 10
BATCH = 128
LEARNINGRATE = 0.001

# %% Fashion MNIST Dataset

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

match_list = ["T-shirt","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle Boot"]

# %% Plot some sample images
plt.figure(figsize=(10,4))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(train_set[i][0].squeeze().numpy(), cmap="gray")
    plt.axis('off')
    plt.title(i)

plt.show()

# %% Dropout model define
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
# %% Batchnormalization define
def batch_normal_init(layer):
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_uniform_(layer.weight)

def BatchNorm_model():
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 20, bias=False),
        nn.BatchNorm1d(20),
        nn.Sigmoid(),
        nn.Linear(20, 20, bias=False),
        nn.BatchNorm1d(20),
        nn.Sigmoid(),
        nn.Linear(20, 20, bias=False),
        nn.BatchNorm1d(20),
        nn.Sigmoid(),
        nn.Linear(20, 10)
    )
    
    model.apply(batch_normal_init)
    
    print(model)
    
    return model

def without_BatchNorm_model():
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 20),
        nn.Sigmoid(),
        nn.Linear(20, 20),
        nn.Sigmoid(),
        nn.Linear(20, 20),
        nn.Sigmoid(),
        nn.Linear(20, 10)
    )
    
    model.apply(batch_normal_init)
    
    print(model)
    
    return model

# %% create model
#model_dropout = Dropout_model()
#model_wo_dropout = without_Dropout_model()

#model_batchnorm = BatchNorm_model()
#model_wo_batchnorm = without_BatchNorm_model()

# Optimizer
#optimizer = optim.Adam(model_dropout.parameters(), lr=LEARNINGRATE)
#optimizer = optim.Adam(model_wo_dropout.parameters(), lr=LEARNINGRATE)

#optimizer = optim.Adam(model_batchnorm.parameters(), lr=LEARNINGRATE)
#optimizer = optim.Adam(model_wo_batchnorm.parameters(), lr=LEARNINGRATE)


# Loss
criterion = nn.CrossEntropyLoss()


# Function to calculate accuracy
def accuracy_fn(Y, out):
    pred = torch.argmax(out, 1)
    correct_prediction = (pred == Y)
    accuracy = correct_prediction.float().mean()
    return accuracy

# %% Checkpoints
import os

checkpoint_dir = 'checkpoints'
model_dir = 'batchnorm_model'
checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# %% Load
def load(model, opimizer, checkpoint_file):
    print("[*] Reading checkpoints...")
    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        print("[*] Success to read checkpoint at epoch {}".format(epoch))
        return True, epoch
    else:
        print("[*] Failed to find a checkpoint")
        return False, 0
    
#%% custom imshow
def custom_imshow(img, i):
    plt.subplot(2, 5, i+1)
    img = img.numpy()
    plt.imshow(np.transpose(img, (1, 2, 0)), cmap="gray")
    
def process():
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        custom_imshow(inputs[0])
        
# %% model choose
model = BatchNorm_model()
optimizer = optim.Adam(model.parameters(), lr=LEARNINGRATE)
# %% Training
fstr = "Iter: {:5}, Loss: {:5.4f}, Accuracy: {:5.4f}"

model.train()
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
    
    checkpoint_file = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pth')
    checkpoint = {
        'epoch':epoch,
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict':optimizer.state_dict(),
    }
    torch.save(checkpoint,checkpoint_file)
    
# %% Load checkpoints
could_load, checkpoint_counter = load(model, optimizer, checkpoint_file)

if could_load:
    start_epoch = (int)(checkpoint_counter)+1
    print("[*] Load SUCCESS")
else:
    start_epoch = 0
    print("[!] Load failed...")
    
EPOCH = 15
for epoch in range(start_epoch, EPOCH):
    for images, labels in train_loader:
        pred = model(images)
        loss = criterion(pred, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    with torch.no_grad():
        train_accuracy = accuracy_fn(labels, F.softmax(model(images), dim=1))
    print(fstr.format(epoch, loss, train_accuracy))
    
    checkpoint_file = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pth')
    checkpoint = {
        'epoch':epoch,
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict':optimizer.state_dict(),
    }
    torch.save(checkpoint,checkpoint_file)
    
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
# %%
print('심헌 201821254')