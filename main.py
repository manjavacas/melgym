#!/usr/bin/python3
import gymnasium as gym
import numpy as np

import melgym

from stable_baselines3.dqn import PPO

INPUT_FILE = './sgs2.inp'
EDF_FILE = './VARIABLES.DAT'

N_EPISODES = 10

N_OBS = 4
N_ACTIONS = 1


def run():

    env = gym.make('sgs-v0', obs_size=N_OBS, n_actions=N_ACTIONS)

    # TRAINING
    # agent = PPO('MlpPolicy', env, verbose=1)
    # agent.learn(total_timesteps=25000)
    # agent.save('ppo_sgs')

    # EVALUATION
    obs, _ = env.reset()
    for e in range(N_EPISODES):
        done = False
        ep_mean_reward = 0.
        n = 0
        while not done:
            action = agent.predict(obs)
            obs, reward, done, _, _ = env.step(action)
            
            print(f'\tTimestep {n}. Reward = {reward}')

            ep_mean_reward = (ep_mean_reward + reward) / n
            n += 1
        
        print(f'Episode {e}. Mean reward = {ep_mean_reward}')
            
if __name__ == '__main__':
    run()
