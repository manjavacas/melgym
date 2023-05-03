#!/usr/bin/python3
import gymnasium as gym
import numpy as np

import random

import melgym

from stable_baselines3.a2c import A2C

INPUT_FILE = './sgs.inp'
EDF_FILE = './VARIABLES.DAT'

N_OBS = 1
N_ACTIONS = 1


def run():

    env = gym.make('sgs-v0', obs_size=N_OBS, n_actions=N_ACTIONS)

    ################## TRAINING ##################
    
    agent = A2C('MlpPolicy', env, verbose=1, n_steps=3, tensorboard_log='./tensorboard/')
    agent.learn(total_timesteps=1000, tb_log_name='sgs')
    agent.save('agent_sgs')

    ################# EVALUATION #################
    
    # obs, _ = env.reset()
    # n = 0
    # while True:
    #     # action = agent.predict(obs)   
    #     action = [random.uniform(10.0, 22.2)]
        
    #     obs, reward, _, _, _ = env.step(action)
        
    #     print(f'\tTimestep {n}. Reward = {reward}')
    #     n += 1

if __name__ == '__main__':
    run()
