# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 21:00:13 2023

@author: USER
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
torch.manual_seed(1)

# %%
EPOCH = 5000
Learning_rate = 0.01
sequence_length = 10

# %% Data
t = np.arange(0, 10 * 2 * np.pi, 0.1)
data = np.sin(t)
plt.plot(data)
plt.show()

idx = np.arange(0, len(data) - sequence_length).reshape((-1, 1))

data_in = np.expand_dims(data[idx + np.arange(sequence_length)], axis = -1)
data_out = data[idx + sequence_length]

input_size = data_in.shape[2]
output_size = data_out.shape[1]

# %% Model
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out[:, -1, :])
        return out, hidden
    
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out[:, -1, :])
        return out, hidden

# model = SimpleRNN(input_size, 3, output_size)
model = LSTM(input_size, 3, output_size)
print(model)

# %% optimizer
optimizer = optim.Adam(model.parameters(), lr=Learning_rate)
mse_loss = nn.MSELoss()

def loss_calc(Y, out):
    loss = mse_loss(out, Y)
    return loss

def train_step(X, Y):
    hidden = None
    model.zero_grad()
    
    out, hidden = model(torch.tensor(X).float(), hidden)
    loss = loss_calc(torch.tensor(Y).float(), out)
    loss.backward()
    optimizer.step()
    
    return loss

# %% Training
for epoch in range(EPOCH):
    noise = 0.1 * np.random.randn(data_in.shape[0], data_in.shape[1], data_in.shape[2])
    loss = train_step(data_in + noise, data_out.reshape(-1,1))
    if epoch % 250 == 0 or epoch == EPOCH - 1:
        print(f"Epoch {epoch:4}, Loss {loss.item():7.4f}")

# %% Prediction
pred_result_pract = np.zeros(len(data_out))
pred_result_ideal = np.zeros(len(data_out))
data_in_cur = data_in[0:1, ...]
with torch.no_grad():
    for i in range(len(data_out)):
        pred_result_pract[i] = model(torch.tensor(data_in_cur).float(), None)[0].numpy().flatten()
        new_data_in = pred_result_pract[i].reshape((1, 1, 1))
        data_in_cur = np.concatenate((data_in_cur[:, 1:, :], new_data_in), axis=1)
        pred_result_ideal[i] = model(torch.tensor(data_in[i:i+1, ...]).float(), None)[0].numpy().flatten()
        
# %% Visualization
plt.figure()
plt.subplot(2,1,1)
plt.plot(data_out)
plt.plot(pred_result_pract)
plt.subplot(2,1,2)
plt.plot(data_out)
plt.plot(pred_result_ideal)

plt.figure()
plt.plot(data_out.flatten() - pred_result_pract)
plt.plot(data_out.flatten() - pred_result_ideal)