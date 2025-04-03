#!/usr/bin/env python3

import melgym
import gymnasium as gym


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
            env.render()


if __name__ == '__main__':
    env = gym.make('pressure')
    rand_control(env)
    env.close()
