#!/usr/bin/python3
import gymnasium as gym
import melgym

INPUT_FILE = './sample.inp'
EDF_FILE = './VARIABLES.DAT'

N_EPISODES = 1

def run():

    env = gym.make('sample-v0', obs_size=1, n_actions=3)
    for i in range(N_EPISODES):
        obs, _ = env.reset()
        print(obs)

    

if __name__ == '__main__':
    run()
