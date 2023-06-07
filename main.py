#!/usr/bin/python3

import melgym

import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO

N_BRANCHES = 1
N_ROOMS = 2


def train(env):
    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log='./tensorboard/')
    model.learn(total_timesteps=25000)
    model.save('./models/model-ppo')


def run(env):
    model = PPO.load('./models/model-ppo')
    obs, _ = env.reset()
    while True:
        action, _ = model.predict(obs)
        obs, reward, _, _, info = env.step(action)
        env.render()
        print(f'\n===========\n{info}\nReward={reward}\n===========\n')


if __name__ == '__main__':
    env = gym.make('hvac-v0', n_obs=N_ROOMS, n_actions=N_BRANCHES,
                   control_horizon=10, max_deviation=20)
    train(env)
    run(env)
    env.close()
