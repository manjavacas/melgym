#!/usr/bin/env python3

import melgym
import gymnasium as gym
import numpy as np


def rbc_controller(env, n_episodes=1):
    """
    Rule-based controller for testing purposes.

    Args:
        env (gym.Env): gymnasium environment.
        n_episodes (int): Number of episodes to run.
    """
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = trunc = False
        while not (done or trunc):
            if obs[0] > 101000.0:
                action = np.array([1.5])
            else:
                action = np.array([.5])
            obs, reward, done, trunc, info = env.step(action)
            print(obs, reward, action, info)
            env.render()

    env.close()


if __name__ == '__main__':
    env = gym.make('pressure')
    rbc_controller(env, n_episodes=1)
    env.close()
