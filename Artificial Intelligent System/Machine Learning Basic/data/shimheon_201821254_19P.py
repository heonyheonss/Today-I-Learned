# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 00:18:05 2023

@author: USER
"""

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import time

import os

torch.manual_seed(42)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# %% Data parameter
BUFFER_SIZE = 6000
BATCH_SIZE = 100
EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16

# %% MNIST Dataset
train_set = datasets.MNIST(
    root = './data/MNIST',
    train = True,
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]), # 데이터를 0에서 255까지 있는 값을 0에서 1사이 값으로 변환
    download = True
)

print('number of training data : ', len(train_set))
print(train_set[0][0].size())

train_loader = torch.utils.data.DataLoader(dataset = train_set, batch_size = BATCH_SIZE, shuffle = True)

original_images, labels = next(iter(train_loader))
print("original_images.shape : ", original_images.detach().numpy().shape)

# %% Plot same sample images
plt.figure(figsize=(10,4))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(train_set[i][0].squeeze().numpy())
    plt.axis('off')
    plt.title(train_set[i][1])

plt.show()

# %% Generator
# Define the generator model
class Generator(nn.Module):
    def __init__(self, noise_dim):
        super(Generator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 7 * 7 * 256, bias = False),
            nn.BatchNorm1d(7 * 7 * 256),
            nn.LeakyReLU(),
            
            nn.Unflatten(1, (256, 7, 7)),
            
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            
            nn.ConvTranspose2d(128, 64,  kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            
            nn.ConvTranspose2d(64, 1,  kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
            nn.Tanh()
        )
        
    def forward(self, x):
        return self.model(x)
    
generator = Generator(noise_dim).to(device)

# %% Discriminator
# Define the descriminator model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size = 5, stride = 2, padding =2),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            
            nn.Conv2d(64, 128, kernel_size = 5, stride = 2, padding =2),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            
            nn.Flatten(),
            nn.Linear(7 * 7 * 128, 1)
        )
    
    def forward(self, x):
        return self.model(x)
    
discriminator = Discriminator().to(device)

# %% Cost Function
criterion = nn.BCEWithLogitsLoss()

def discriminator_loss(real_output, fake_output):
    real_loss = criterion(real_output, torch.ones_like(real_output))
    fake_loss = criterion(fake_output, torch.zeros_like(fake_output))
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    gen_loss = criterion(fake_output, torch.ones_like(fake_output))
    return gen_loss

generator_optimizer = optim.Adam(generator.parameters(), lr=1e-4)
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=1e-4)

# %% 
def train_step(images):
    noise = torch.randn(BATCH_SIZE, noise_dim).to(device)
    images = images.to(device)
    
    # Generator training
    generator.zero_grad()
    generated_images = generator(noise)
    fake_output = discriminator(generated_images)
    gen_loss = generator_loss(fake_output)
    gen_loss.backward()
    generator_optimizer.step()
    
    # Discriminator training
    discriminator.zero_grad()
    real_output = discriminator(images)
    fake_output = discriminator(generated_images.detach())
    disc_loss = discriminator_loss(real_output, fake_output)
    disc_loss.backward()
    discriminator_optimizer.step()
    
    return gen_loss.item(), disc_loss.item()

# %%
seed = torch.randn([num_examples_to_generate, noise_dim]).to(device)

def train(dataloader, epochs):
    start = time.time()
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(dataloader):
            train_step(images)
            print('\rep{:4}/{} : {:-3}/{:3}'.format(epoch + 1, epochs, i + 1, len(dataloader)), end='')
            
            generator_images(generator, seed)
            
            print('\rTime for epoch {:3} is {:.2f} min'.format(epoch + 1, (time.time() - start)/60))
            
def generator_images(model, test_input):
    
    predictions = model(test_input).to('cpu')
    print(predictions.shape)
    
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(to_pil_image(0.5 * predictions[i,:,:,:] + 0.5), cmap='gray')
        plt.axis('off')
    plt.show()
    
# %%
train(train_loader, EPOCHS)
print('심헌 201821254')
