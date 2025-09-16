# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 15:32:27 2023

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
# %% Conv2D default initialize
layer = torch.nn.Conv2d(1, 1, 2)
# %%
print(layer.weight.data)