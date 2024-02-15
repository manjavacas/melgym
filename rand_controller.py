#!/usr/bin/env python

import melgym
import gymnasium as gym
import numpy as np

from melgym.utils.aux import summary


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
        n_steps = 1
        while not done and not truncated:
            env.render()
            action = env.action_space.sample()
            obs, reward, truncated, done, info = env.step(action)
            summary(n_steps, action, obs, reward, info)
            n_steps += 1


if __name__ == '__main__':
    env = gym.make('branch_1', render_mode='pressures', time_bt_frames=.01)
    rand_control(env)
    env.close()
