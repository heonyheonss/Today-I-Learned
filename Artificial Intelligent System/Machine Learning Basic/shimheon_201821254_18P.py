# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 22:56:24 2023

@author: USER
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import matplotlib.pyplot as plt
import gym

from IPython import display
import os

# %% Initialize Environment & Initialize Qnet
env = gym.make('Pendulum-v1', render_mode="rgb_array")
class EnvMod:
    def __init__(self, env):
        self.env = env
        self.state = []
        
    def reset(self):
        state, info = self.env.reset(seed=42)
        self.state = torch.tensor(state, dtype=torch.float32).view(1, -1)
        return self.state.clone()
    
    def step(self, action):
        state_, reward, done, truncated, info = self.env.step(action)
        state_ = torch.tensor(state_, dtype=torch.float32).view(1, -1)
        reward = reward - 0.5 * torch.abs(torch.tensor(action[0], dtype=torch.float32))
        self.state = state_.clone()
        return state_, reward, done, truncated, info
    
    def render(self):
        return self.env.render()
    
myenv = EnvMod(env)
act_spc = list(np.array([[-2., -1.6, -0.8, -0.4, -0.2, -0.1, 0., 0.1, 0.2, 0.4, 0.8, 1.6, 2.]]).T) # 총 13개의 act

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


def s_normalizer(state):
    state_n = (state - torch.tensor([[-1., -1., -8.]], 
                                    dtype = torch.float32))/torch.tensor([[2.,2.,16.]], dtype=torch.float32)
    return state_n

def obs_func(state):
    return state

# %% Define parameters
EPISODES = 30
EPISODES_LENGTH = 500
BATCH = 100
discount = 0.98
target_update_freq = 100

state = myenv.reset()
State_SZ = state.shape[1]
Action_SZ = len(act_spc)

# %% Define function for training
# Instantiate Q-networks
qnet = Qnet(State_SZ, Action_SZ)
qnet_target = Qnet(State_SZ, Action_SZ)
qnet_target.load_state_dict(qnet.state_dict())

# Optimizer
opt = optim.Adam(qnet.parameters(), lr=0.001)

# Memory
mem_idx = 0
mem_len_cur = 0
mem_len = 1000
mem = {
       'state': torch.zeros((mem_len, State_SZ)),
       'action': torch.zeros((mem_len,)),
       'state_': torch.zeros((mem_len, State_SZ)),
       'reward': torch.zeros((mem_len,)),
       'done': torch.zeros((mem_len,)),
       }

# Get action function
def get_action(state, eps, training=True):
    sn = obs_func(s_normalizer(state))
    if training:
        rnd_val = np.random.random()
        if rnd_val < eps:
            a_idx = random.sample(range(len(act_spc)), 1)
        else:
            qvals = qnet(sn).detach().cpu().numpy()
            a_idx = np.argmax(qvals, -1).astype(np.int32)
    else:
        qvals = qnet(sn).detach().cpu().numpy()
        a_idx = np.argmax(qvals, -1).astype(np.int32)  
    return a_idx

# %% Training DQN
loss_mv = 0.
target_update_step = 0

for e in range(EPISODES):
    state = myenv.reset()
    eps = max(1.0 / (1. + 0.05 * e), 0.05)
    q_update_num = 0
    return_val = 0.
    
    state_vec = np.nan*np.zeros((EPISODES_LENGTH, 3)).astype(np.float32)
    action_vec = np.nan*np.zeros((EPISODES_LENGTH, 1)).astype(np.float32)
    
    for i in range(EPISODES_LENGTH):
        a_idx = get_action(state, eps, training=True)
        a = act_spc[a_idx[0]]
        state_, reward, done, truncated, info = myenv.step(a)
        
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

print()

# Plot Episode
ax = plt.subplot(3,1,1)
plt.plot(180/np.pi*np.arctan2(state_vec[:,1], state_vec[:, 0]), '.-')
plt.plot([0,199],[0,0],'r-')
axins = ax.inset_axes([0.9, 0.75, 0.1, 0.45])
axins.plot(180/np.pi*np.arctan2(state_vec[:,1], state_vec[:, 0]), '.-')
plt.plot([0,199],[0,0],'r-')
axins.set_ylim(-10, 10)
axins.set_xlim(200, 200)

plt.subplot(3,1,2)
plt.plot(180/np.pi*state_vec[:,2],'.-')

plt.subplot(3,1,3)
plt.plot(action_vec[:,0],'.-')
plt.show()

# %% save initial q
if not os.path.exists('saveq'):
    os.makedirs('saveq')
    
torch.save(qnet.state_dict(), 'saveq/save01.pth')

# %% Testing DQN
# qnet.load_state_dict(torch.load('saveq/save01.pth'))

myenv.env.unwrapped.render_mode = 'human'

eps = 0.
for e in range(3):
    state = myenv.reset()
    
    state_vec = np.nan * np.zeros((EPISODES_LENGTH, 3)).astype(np.float32)
    action_vec = np.nan * np.zeros((EPISODES_LENGTH, 1)).astype(np.float32)
    return_val = 0.
    
    for i in range(EPISODES_LENGTH):
        a_idx = get_action(state, eps, training=False)
        a = act_spc[a_idx[0]]
        
        state_, reward, done, truncated, info = myenv.step(a)
        
        state_vec[i,:] = state
        action_vec[i,:] = act_spc[a_idx[0]]
        
        return_val = return_val + reward
        
        print('\rep{:05}|eps{:4.2f}|loss{:12.7f}|qu{:03}|rsum{:12.3f}'.format(
            e,
            0.,
            0.,
            0,
            return_val), end='')        
        
        state = state_
        
        if done:
            break
    
    print()
    
    ax = plt.subplot(3,1,1)
    plt.plot(180/np.pi*np.arctan2(state_vec[:,1], state_vec[:, 0]), '.-')
    plt.plot([0,199],[0,0],'r-')
    axins = ax.inset_axes([0.9, 0.75, 0.1, 0.45])
    axins.plot(180/np.pi*np.arctan2(state_vec[:,1], state_vec[:, 0]), '.-')
    plt.plot([0,199],[0,0],'r-')
    axins.set_ylim(-10, 10)
    axins.set_xlim(200, 200)

    plt.subplot(3,1,2)
    plt.plot(180/np.pi*state_vec[:,2],'.-')

    plt.subplot(3,1,3)
    plt.plot(action_vec[:,0],'.-')
    plt.show()

env.close()
print('201821254 심헌')