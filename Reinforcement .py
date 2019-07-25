#!/usr/bin/env python
# coding: utf-8

# In[31]:


get_ipython().system('pip install gym')
import tensorflow as tf
import numpy as np
import gym


# In[32]:


import matplotlib.pyplot as plt
from gym.envs.registration import register
import random as pr


# # FrozenLake

# In[33]:


def rargmax(vector):
    m=np.amax(vector)
    indices=np.nonzero(vector==m)[0]
    return pr.choice(indices)

#register(
    id="FrozenLake-v0",
    entry_point="gym.envs.toy_text:FrozenLakeEnv",
    kwargs={"map_name":"4x4",
           "is_slippery":"False"}


env=gym.make("FrozenLake-v0")

Q=np.zeros([env.observation_space.n, env.action_space.n])

num_episodes=200

rList=[]
for i in range(num_episodes):
    state=env.reset()
    rAll=0
    done=False
    
    while not done:
        action=rargmax(Q[state,:])
        
        new_state,reward,done,_=env.step(action)
        Q[state,action]= reward+np.max(Q[state,:])
        
        rAll+=reward
        state=new_state
        
        rList.append(rAll)
        
print("Success rate:" + str(sum(rList)/num_episodes))
print("Final Q-table values")
print("left fown right up")
print(Q)

plt.bar(range(len(rList)),rList,color="blue")
plt.show()


# # Taxi

# In[34]:


import gym
#env=gym.make("Taxi-v2")
observation=env.reset()

for _ in range(1000):
    env.render()
    action=env.action_space.sample()
    observation, reward,done,info=env.step(action)


# # Open AI Game

# In[43]:


pip install --no-index -f https://github.com/Kojoley/atari-py/releases atari_py


# In[ ]:




