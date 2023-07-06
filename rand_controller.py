#!/usr/bin/python3

import numpy as np
import random
import melgym
import gymnasium as gym

from melgym.utils.aux import summary


def rand_control(env):
    obs, _ = env.reset()
    done = False
    truncated = False
    n_episode = 1
    while not done and not truncated:
        action = env.action_space.sample()
        # action = np.array([-0.85])
        obs, reward, truncated, done, info = env.step(action)
        summary(n_episode, action, obs, reward, info)
        env.render()
        n_episode += 1


if __name__ == '__main__':
    env = gym.make('simple-v0', render_mode='pressures')

    rand_control(env)
    env.close()
