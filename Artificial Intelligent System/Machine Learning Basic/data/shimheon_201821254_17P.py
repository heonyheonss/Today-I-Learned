# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 14:57:12 2023

@author: heon
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import gym

from IPython import display
import matplotlib.pyplot as plt

# %%
env = gym.make('FrozenLake-v1', render_mode="rgb_array", is_slippery=False)
state_spc = torch.eye(16)
act_spc = [0, 1, 2, 3]

def s_normalizer(state):
    return state

def obs_func(state):
    return state

class EnvMod:
    def __init__(self, env):
        self.env = env
        self.state = []
        
    def reset(self):
        state, _ = self.env.reset()
        self.state = state_spc[state, :].reshape([1, -1]).float()
        return obs_func(s_normalizer(self.state)).clone()
    
    def step(self, action):
        state_idx, reward, done, info, _ = self.env.step(action)
        state_ = state_spc[state_idx, :].reshape([1, -1]).float()
        return obs_func(s_normalizer(state_)), reward, done, info
    
    def render(self):
        return self.env.render()
    
myenv = EnvMod(env)

# %% Initialize Qnet & replay memory
leaky_relu = nn.LeakyReLU(0.2)

class Qnet(nn.Module):
    def __init__(self, state_size, action_size):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(state_size, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, action_size)
        
    def forward(self, x):
        x = leaky_relu(self.fc1(x))
        x = leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
# %%
def get_action(state, eps, training=True):
    sn = state
    if training:
        rnd_val = random.random()
        if rnd_val < eps:
            a_idx = random.sample(range(len(act_spc)), 1)
        else:
            qvals = qnet(sn).detach().numpy()
            a_idx = torch.argmax(torch.tensor(qvals), dim=-1).int().numpy()
    else:
        qvals = qnet(sn).detach().numpy()
        a_idx = torch.argmax(torch.tensor(qvals), dim=-1).int().numpy()    
    return a_idx
    
# %% Instantiate Q-networks
state_size = state_spc.shape[1]
action_size = len(act_spc)
qnet = Qnet(state_size, action_size)
qnet_target = Qnet(state_size, action_size)
qnet_target.load_state_dict(qnet.state_dict())

opt = optim.Adam(qnet.parameters(), lr=0.001)

mem_idx = 0
mem_len_cur = 0
mem_len = 100
mem = {
       'state': torch.zeros((mem_len, state_size)),
       'action': torch.zeros((mem_len,)),
       'state_': torch.zeros((mem_len, state_size)),
       'reward': torch.zeros((mem_len,)),
       'done': torch.zeros((mem_len,)),
       }

# %%
EPISODES = 500
BATCH = 10
discount = 0.98
target_update_freq = 20

# %%
loss_mv = 0.
target_update_step = 0
q_update_num = 0
for e in range(EPISODES):
    state = myenv.reset()
    eps = max(1.0 / (1. + 0.05 * e), 0.05)
    return_val = 0.
    while True:
        a_idx = get_action(state, eps, training=True)
        a = act_spc[a_idx[0]]
        state_, reward, done, info = myenv.step(a)
        
        mem['state'][mem_idx, :] = state
        mem['action'][mem_idx] = a_idx[0]
        mem['state_'][mem_idx, :] = state_
        mem['reward'][mem_idx] = reward
        mem['done'][mem_idx] = done
        mem_idx = (mem_idx+1) % mem_len
        mem_len_cur = min(mem_len_cur + 1, mem_len)
        
        mem_batch_idx = torch.tensor(random.sample(range(mem_len_cur), k = min(BATCH, mem_len_cur)))
        
        sBtc = mem['state'][mem_batch_idx, :]
        aBtc = mem['action'][mem_batch_idx].long()
        s_Btc = mem['state_'][mem_batch_idx, :]
        rBtc = mem['reward'][mem_batch_idx]
        dBtc = mem['done'][mem_batch_idx]
        
        sn = obs_func(s_normalizer(sBtc))
        s_n = obs_func(s_normalizer(s_Btc))
        qvals = qnet(s_n).detach().numpy()
        idx_max = torch.argmax(torch.tensor(qvals), dim=-1)
        y = rBtc + (1 - dBtc) * discount * qnet_target(s_n).detach().numpy()[torch.arange(len(idx_max)), idx_max]
        
        opt.zero_grad()
        q = torch.sum(qnet(sn) * nn.functional.one_hot(aBtc, len(act_spc)).float(), dim=-1)
        loss = nn.MSELoss()(y.clone().detach(), q)
        loss.backward()
        opt.step()
        
        target_update_step = target_update_step + 1
        if target_update_step >= target_update_freq:
            qnet_target.load_state_dict(qnet.state_dict())
            target_update_step = 0
            q_update_num = q_update_num + 1
        return_val = return_val + reward
        
        loss_mv = loss_mv * 0.98 + loss.item() * 0.02
        print('\rep{:05}|eps{:4.2f}|loss{:12.7f}|qu{:03}|rsum{:12.3f}'.format(
            e,
            eps,
            loss_mv,
            q_update_num,
            return_val), end='')
        
        state = state_
        
        if done:
            break
 
# %% test
state = myenv.reset()
while True:
    a_idx = get_action(state, eps, training=False)
    a = act_spc[a_idx[0]]
    
    screen = env.render()
    plt.imshow(screen)
    
    display.clear_output(wait=True)
    display.display(plt.gcf())
    
    state_, reward, done, info = myenv.step(a)
    
    state = state_
    
    if done:
        break

myenv.render()

# %%
str_f = ['f',]
str_grid = np.array([str_f[0] for i in range(16)])
str_grid[0] = 's'
str_grid[15] = 'g'
str_grid[5] = 'h'
str_grid[7] = 'h'
str_grid[11] = 'h'
str_grid[12] = 'h'
str_list = []
for i in range(16):
    qvals = qnet(torch.eye(16)[[i], :])
    
    if str_grid[i] == 'h' or str_grid[i] == 'g':
        str_list.append('     -     ')
        str_list.append(('  -  ['+str_grid[i]+']  -  '))
        str_list.append('     -     ')
    else:
        str_list.append('{:9.2f}   '.format(qvals[0,3]))
        str_list.append(('{:5.2f}['+str_grid[i]+']{:5.2f}').format(qvals[0,0], qvals[0,2]))
        str_list.append('{:9.2f}   '.format(qvals[0,1]))
str_list = np.array(str_list)
print('---------------------------------------------------------')
for i in range(4):
    j = i * 12
    print('|'+str_list[j  ]+'|'+str_list[j+3]+'|'+str_list[j+6]+'|'+str_list[j+9 ]+'|')
    print('|'+str_list[j+1]+'|'+str_list[j+4]+'|'+str_list[j+7]+'|'+str_list[j+10]+'|')
    print('|'+str_list[j+2]+'|'+str_list[j+5]+'|'+str_list[j+8]+'|'+str_list[j+11]+'|')
    if i<3:
        print('---------------------------------------------------------')
print('---------------------------------------------------------')
print('심헌 201821254')
