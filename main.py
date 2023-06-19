#!/usr/bin/python3

import melgym

import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO, SAC, DQN
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

from datetime import datetime

N_BRANCHES = 1
N_ROOMS = 2

MAX_DEVIATION = 20
MAX_VEL = 1
CONTROL_HORIZON = 5

TRAIN_TIMESTEPS = 100_000
EVAL_FREQ = 5_000
EVAL_EPISODES = 1


class MetricsCallback(BaseCallback):
    """
    Custom callback for plotting additional simulation data in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        self.logger.record(
            'sim_data/Velocity', self.training_env.get_attr('last_velocity')[0])
        self.logger.record(
            'sim_data/Truncated', self.training_env.get_attr('last_truncated')[0])
        # for cv, pressure in self.training_env.get_attr('last_pressures')[0].items():
        #     self.logger.record('sim_data/Pressure-' + cv, pressure)
        # for cv, distance in self.training_env.get_attr('last_distances')[0].items():
        #     self.logger.record('sim_data/Distance-' + cv, distance)

        return True


def train(env):

    # Evaluation environment
    env_eval = gym.make('simple-v0', n_obs=N_ROOMS, n_actions=N_BRANCHES,
                        control_horizon=CONTROL_HORIZON, max_deviation=MAX_DEVIATION, max_vel=MAX_VEL)

    # Callbacks
    eval_callback = EvalCallback(
        env_eval, best_model_save_path='./best_models', eval_freq=EVAL_FREQ, n_eval_episodes=EVAL_EPISODES)

    model = PPO('MlpPolicy', env, verbose=1,
                tensorboard_log='./tensorboard')
    model.learn(total_timesteps=TRAIN_TIMESTEPS,
                progress_bar=True, callback=[eval_callback, MetricsCallback()])


def run(env, model_id='best_model'):
    model = PPO.load('best_models/' + model_id)
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
    train(env)
    run(env)
    env.close()
