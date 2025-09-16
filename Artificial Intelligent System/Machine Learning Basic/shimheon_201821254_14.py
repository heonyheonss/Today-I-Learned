# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 15:12:48 2023

@author: heon
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
torch.manual_seed(1)
# %%
EPOCH = 300
Learning_rate = 0.001

# %% Data
idx2word = ['<bos>', '<eos>', 'e', 'h', 'i', 'm', 'n', 'o', 's']
word2idx = {word: i for i, word in enumerate(idx2word)}

seq = ['<bos>', 's', 'h', 'i', 'm', 'h', 'e', 'o', 'n', '<eos>']
seq_idx = [word2idx[word] for word in seq]
data = np.eye(len(idx2word))[seq_idx]
data_in = data[0 : len(seq_idx) - 1].reshape(1, len(seq_idx) - 1, len(idx2word))
data_out = data[1 : len(seq_idx)].reshape(1, len(seq_idx) - 1, len(idx2word))

# %% Model
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out, hidden
    
    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)
    
model = SimpleRNN(len(idx2word), len(idx2word), len(idx2word))
print(model)

# %% optimizer
optimizer = optim.Adam(model.parameters(), lr=Learning_rate)
ce = nn.CrossEntropyLoss()

def loss_calc(Y, out):
    Y = torch.argmax(Y, dim=2)
    loss = ce(out.permute(0, 2, 1), Y)
    return loss

def accuracy_fn(Y, out):
    Y_chk = torch.argmax(Y, dim=2)
    out_chk = torch.argmax(out, dim=2)
    chk = torch.eq(Y_chk, out_chk).float()
    accuracy = torch.mean(chk)
    return accuracy.item()

def train_step(X, Y):
    hidden = model.init_hidden(X.shape[0])
    model.zero_grad()
    
    out, _ = model(X, hidden)
    loss = loss_calc(Y, out)
    loss.backward()
    optimizer.step()
    
    return loss, accuracy_fn(Y, out)

# %% Training
for epoch in range(EPOCH):
    loss, acc = train_step(torch.tensor(data_in).float(), torch.tensor(data_out).float())
    if epoch % 50 == 0 or epoch == EPOCH - 1:
        print(f"Epoch {epoch:4}, Loss {loss.item():7.4f}, Accuracy {acc:5.2f}")
        
# %% Prediction
def prediction(in_words):
    in_idx = [word2idx[word] for word in in_words]
    word_onehot = np.eye(len(idx2word))[in_idx].reshape(1, len(in_idx), len(idx2word))
    out_onehot, _ = model(torch.tensor(word_onehot).float(), model.init_hidden(1))
    out_idx = np.argmax(out_onehot.detach().numpy(), axis=2).flatten()
    out_words = [idx2word[idx] for idx in out_idx]
    return out_words

in_words_all = ['<bos>', 's', 'h', 'i', 'm', 'h', 'e', 'o', 'n']
for i in range(len(in_words_all)):
    in_words = in_words_all[0 : i+1]
    out_words = prediction(in_words)
    print('in:', in_words[-1], '-> out:', out_words[-1])
# %%
print("201821254 심헌")