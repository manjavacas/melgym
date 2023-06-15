#!/usr/bin/python3

import melgym

import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback

from datetime import datetime

N_BRANCHES = 1
N_ROOMS = 2

MAX_DEVIATION = 20
MAX_VEL = 1
CONTROL_HORIZON = 5


def train(env):

    env_eval = gym.make('simple-v0', n_obs=N_ROOMS, n_actions=N_BRANCHES,
                        control_horizon=CONTROL_HORIZON, max_deviation=MAX_DEVIATION, max_vel=MAX_VEL)

    # Callbacks
    eval_callback = EvalCallback(
        env_eval, best_model_save_path='./best_models', eval_freq=5000, n_eval_episodes=1)

    model = PPO('MlpPolicy', env, verbose=1,
                tensorboard_log='./tensorboard')
    model.learn(total_timesteps=100000,
                progress_bar=True, callback=[eval_callback])


def run(env):
    model = PPO.load('best_models/PPO-discrete')
    obs, _ = env.reset()
    done = False
    truncated = False
    while not done and not truncated:
        action, _ = model.predict(obs)
        obs, reward, truncated, done, info = env.step(action)
        env.render()
        print(f'\n===========\n{info}\nReward={reward}\n===========\n')


if __name__ == '__main__':
    env = gym.make('simple-v0', n_obs=N_ROOMS, n_actions=N_BRANCHES,
                   control_horizon=CONTROL_HORIZON, max_deviation=MAX_DEVIATION, max_vel=MAX_VEL)
    #train(env)
    run(env)
    env.close()
