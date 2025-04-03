#!/usr/bin/env python3

import melgym
import gymnasium as gym
import numpy as np


N_TEST_EPISODES = 1


def rand_control(env):
    """
    Random controller for testing purposes.

    Args:
        env (gym.Env): gymnasium environment.
    """
    for _ in range(N_TEST_EPISODES):
        obs, _ = env.reset()
        done = False
        truncated = False
        while not (done or truncated):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            print(info, reward, done, truncated)
            env.render()

    env.close()


if __name__ == '__main__':
    env = gym.make('pressure')
    rand_control(env)
    env.close()
