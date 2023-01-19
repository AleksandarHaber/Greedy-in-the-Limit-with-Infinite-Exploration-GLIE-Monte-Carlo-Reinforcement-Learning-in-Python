
'''
Python Implementation of 
the Greedy in the Limit with Infinite Exploration (GLIE) Monte Carlo Control Method

Author: Aleksandar Haber

'''
# Note: 
# You can either use gym (not maintained anymore) or gymnasium (maintained version of gym)    
    
# tested on     
# gym==0.26.2
# gym-notices==0.0.8

#gymnasium==0.27.0
#gymnasium-notices==0.0.1

# classical gym 
import gym
# instead of gym, import gymnasium 
# import gymnasium as gym

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
from functions import MonteCarloControlGLIE
 
# create the environment 
# is_slippery=False, this is a completely deterministic environment, 
# uncomment this if you want to render the environment during the solution process
# however, this will slow down the solution process
#env=gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False,render_mode="human")

# here we do not render the environment for speed purposes
env=gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)
env.reset()
# render the environment
# uncomment this if you want to render the environment
#env.render()
# this is used to close the rendered environment
#env.close()
 
# investigate the environment
# observation space - states 
env.observation_space
 
env.action_space
# actions:
#0: LEFT
#1: DOWN
#2: RIGHT
#3: UP
 
# general parameters for the Monte Carlo method
# select the discount rate
discountRate=0.9
# number of states - determined by the Frozen Lake environment
stateNumber=16
# number of possible actions in every state - determined by the Frozen Lake environment
actionNumber=4
# maximal number of iterations of the policy iteration algorithm 
numberOfEpisodes=10000
# initial value of epsilon
initialEpsilon=0.2
# learn the optimal policy
finalLearnedPolicy,Qmatrix=MonteCarloControlGLIE(env,stateNumber,numberOfEpisodes,discountRate,initialEpsilon)

# to interpret the final learned policy you need this information
# actions: 0: LEFT, 1: DOWN, 2: RIGHT, 3: UP
# let us simulate the learned policy
# this will reset the environment and return the agent to the initial state
env=gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False,render_mode='human')
(currentState,prob)=env.reset()
env.render()
time.sleep(2)
# since the initial state is not a terminal state, set this flag to false
terminalState=False
for i in range(100):
    # here we step and return the state, reward, and boolean denoting if the state is a terminal state
    if not terminalState:
        (currentState, currentReward, terminalState,_,_) = env.step(int(finalLearnedPolicy[currentState]))
        time.sleep(1)
    else:
        break
time.sleep(2)
env.close()



























     