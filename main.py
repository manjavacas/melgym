#!/usr/bin/python3

import melgym

import gymnasium as gym
import numpy as np

N_BRANCHES = 1
N_ROOMS = 2


def run():
    env = gym.make('hvac-v0', n_obs=N_ROOMS, n_actions=N_BRANCHES)
    obs, _ = env.reset()

    while True:
        print(env.step(np.random.uniform(0, 10, 1)))

    # env.close()


if __name__ == '__main__':
    run()
