#!/usr/bin/python3

import melgym

import gymnasium as gym
import numpy as np

N_BRANCHES = 1
N_ROOMS = 2


def run():
    env = gym.make('hvac-v0', n_obs=N_ROOMS, n_actions=N_BRANCHES)
    obs, info = env.reset()
    terminated = False
    truncated = False
    while not terminated and not truncated:
        env.render()
        obs, reward, truncated, terminated, info = env.step(env.action_space.sample())
        print(f'info={info}\nreward={reward}, obs={obs}, terminated={terminated}, truncated={truncated}\n==============\n')
    # env.close()


if __name__ == '__main__':
    run()
