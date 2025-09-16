# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 15:26:14 2023

@author: heon
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

np.random.seed(1)
torch.manual_seed(1)
# %%
EPOCH = 1000
learning_rate = 0.005
#%%
idx2word_kr = ['<pad>', '<bos>', '<eos>', '나는', '너는', '그는', '인공지능을',\
               '공부한다', '공부하지', '않는다']
word2idx_kr = {word:i for i, word in enumerate(idx2word_kr)}

idx2word_en = ['<pad>', '<bos>', '<eos>', 'I', 'You', 'He', 'study', 'studies',\
               'do', 'does', 'not', 'an', 'AI']
word2idx_en = {word:i for i, word in enumerate(idx2word_en)}

seq_kr = []
seq_en = []

seq_kr.append(['<bos>', '나는', '인공지능을', '공부한다', '<eos>'])
seq_kr.append(['<bos>', '너는', '인공지능을', '공부한다', '<eos>'])
seq_kr.append(['<bos>', '그는', '인공지능을', '공부한다', '<eos>'])
seq_kr.append(['<bos>', '나는', '인공지능을', '공부하지', '않는다', '<eos>'])
seq_kr.append(['<bos>', '너는', '인공지능을', '공부하지', '않는다', '<eos>'])
seq_kr.append(['<bos>', '그는', '인공지능을', '공부하지', '않는다', '<eos>'])

seq_en.append(['<bos>', 'I', 'study', 'an', 'AI', '<eos>'])
seq_en.append(['<bos>', 'You', 'study', 'an', 'AI', '<eos>'])
seq_en.append(['<bos>', 'He', 'studies', 'an', 'AI', '<eos>'])
seq_en.append(['<bos>', 'I', 'do', 'not', 'study', 'an', 'AI', '<eos>'])
seq_en.append(['<bos>', 'You', 'do', 'not', 'study', 'an', 'AI', '<eos>'])
seq_en.append(['<bos>', 'He', 'does', 'not', 'study', 'an', 'AI', '<eos>'])

# %%
seq_kr_idx = [[word2idx_kr[word] for word in seq] for seq in seq_kr]
seq_en_idx = [[word2idx_en[word] for word in seq] for seq in seq_en]

maxlen_kr = max(map(len, seq_kr_idx))
maxlen_en = max(map(len, seq_en_idx))
maxlen = maxlen_kr + maxlen_en

seq_kr_idx_wpad = [x if len(x) == maxlen_kr else [0] * (maxlen_kr - len(x)) + x for x in seq_kr_idx]
seq_en_idx_wpad = [x if len(x) == maxlen_en else [0] * (maxlen_en - len(x)) + x for x in seq_en_idx]

seq_kr_idx_wpad = [x if len(x) == maxlen else [0] * (maxlen - len(x)) + x for x in seq_kr_idx_wpad]
seq_en_idx_wpad = [x if len(x) == maxlen else [0] * (maxlen - len(x)) + x for x in seq_en_idx_wpad]

data_kr = F.one_hot(torch.tensor(seq_kr_idx_wpad), num_classes=len(idx2word_kr)).float()
data_en = F.one_hot(torch.tensor(seq_en_idx_wpad), num_classes=len(idx2word_en)).float()

print('data_kr.shape: ', data_kr.shape)
print('data_en.shape: ', data_en.shape)

# %%
class BasicModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=15):
        super(BasicModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x
    
class StackedModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=15):
        super(StackedModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.linear(x)
        return x

class BidirectionalStackModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=15):
        super(BidirectionalStackModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size= 2 * hidden_size, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(2 * hidden_size, output_size)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.linear(x)
        return x

model = StackedModel(data_kr.shape[2], data_en.shape[2])

# %%
optimizer = optim.Adam(model.parameters(), learning_rate)

ce_cone = nn.CrossEntropyLoss(reduction = 'none')

def loss_calc(Y, out):
    mask = (torch.argmax(Y, dim=2) != 0).float()
    loss = torch.mean(ce_cone(out.permute(0, 2, 1), torch.argmax(Y, dim=2)) * mask)
    return loss

def accuracy_fn(Y, out):
    Y_chk = torch.argmax(Y, dim=2)
    out_chk = torch.argmax(out, dim=2)
    chk = torch.eq(Y_chk, out_chk).float()
    mask = (torch.argmax(Y, dim=2) != 0).float()
    accuracy = torch.sum(chk * mask) / torch.sum(chk)
    return accuracy

def train_step(X, Y):    
    out = model(X)
    loss = loss_calc(Y, out)
    
    model.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss, accuracy_fn(Y, out)

# %% Training
for epoch in range(EPOCH):
    loss, acc = train_step(data_kr, data_en)
    if epoch % 250 == 0 or epoch == EPOCH - 1:
        print("Epoch {:4}, Loss {:7.4f}, Accuracy {:5.2f}".format(epoch, loss.item(), acc.item()))
        
# %%
for i in range(6):
    idx_in = data_kr[i].argmax(dim=1)[5:]
    print([idx2word_kr[idx] for idx in idx_in if idx > 2], end = ' -> ')
    idx_out = model(data_kr[[i], ...]).squeeze().argmax(dim=1)[5:]
    print([idx2word_en[idx] for idx in idx_out if idx > 2])
    
# %%
print('심헌 201821254')