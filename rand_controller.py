#!/usr/bin/env python

import melgym

import numpy as np
import gymnasium as gym

from melgym.utils.aux import summary


def rand_control(env):
    obs, _ = env.reset()
    done = False
    truncated = False
    n_steps = 1
    while not done and not truncated or n_steps < 500:
        action = env.action_space.sample()
        # action = np.array([-0.842])
        obs, reward, truncated, done, info = env.step(action)
        summary(n_steps, action, obs, reward, info)
        # env.render()
        n_steps += 1


if __name__ == '__main__':
    env = gym.make('simple-v0', render_mode='distances', time_bt_frames=.5)

    rand_control(env)
    env.close()
