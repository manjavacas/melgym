#!/usr/bin/env python3

import melgym
import gymnasium as gym

import numpy as np


def rand_control(env, n_episodes=1):
    """
    Random controller for testing purposes.

    Args:
        env (gym.Env): gymnasium environment.
        n_episodes (int): Number of episodes to run.
    """
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = trunc = False
        while not (done or trunc):
            action = env.action_space.sample()
            obs, reward, done, trunc, info = env.step(action)

    env.close()


if __name__ == '__main__':
    env = gym.make('pressure', render_mode='rgb_array')
    rand_control(env, n_episodes=100)
    env.close()
