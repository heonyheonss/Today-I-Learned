# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 19:46:24 2023

@author: heon
"""

import numpy as np
import random
import gym
from IPython import display
import matplotlib.pyplot as plt

#env = gym.make('FrozenLake-v1', render_mode="rgb_array", is_slippery=False)
env = gym.make('FrozenLake-v1', render_mode="human", is_slippery=False)
act_list = [0, 1, 2, 3]

for _ in range(5):
    env.reset()
    t = 0
    while True:
        t += 1
        
        #screen = env.render()
        #plt.imshow(screen)
        
        #display.clear_output(wait = True)
        #display.display(plt.gcf())
        
        act = act_list[np.random.randint(0, len(act_list))]
        observation, reward, done, info, _ = env.step(act)
        
        if done:
            print("Episode finished afte {} timesteps".format(t))
            break
        
#plt.close()
env.close()

print("심헌 201821254")