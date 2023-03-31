#!/usr/bin/python3
import gymnasium as gym
import melgym

from stable_baselines3.dqn import DQN

INPUT_FILE = './sample.inp'
EDF_FILE = './VARIABLES.DAT'

N_EPISODES = 1

N_CVS = 1
N_ACTIONS = 7

def run():

    env = gym.make('sample-v0', obs_size=N_CVS, n_actions=N_ACTIONS)

    agent = DQN(env)

    for i in range(N_EPISODES):
        obs, _ = env.reset()
        # (en main) [P1, P2, P3... Pk] -> NN -> [action1, action2, action3... actionN]
        actions = agent.predict(obs)
        next_obs, reward, done, _, _ = env.step(actions)
    

if __name__ == '__main__':
    run()
