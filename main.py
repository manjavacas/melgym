#!/usr/bin/python3

import melgym

import gymnasium as gym
import numpy as np

import copy

from stable_baselines3 import PPO, SAC, DQN
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback


MAX_DEVIATION = 20_000
CHECK_DONE_TIME = 1_000
CONTROL_HORIZON = 5

TRAIN_TIMESTEPS = 300_000
EVAL_FREQ = 10_000
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
        return True


def train(env):

    # Evaluation environment
    env_eval = copy.deepcopy(env)

    # Callbacks
    eval_callback = EvalCallback(
        env_eval, best_model_save_path='./best_models/', eval_freq=EVAL_FREQ, n_eval_episodes=EVAL_EPISODES)

    model = PPO('MlpPolicy', env, verbose=1,
                tensorboard_log='./tensorboard/')
    model.learn(total_timesteps=TRAIN_TIMESTEPS,
                progress_bar=True, callback=[eval_callback, MetricsCallback()])


def run(env, model_id='best_model'):
    model = PPO.load('./best_models/' + model_id)
    obs, _ = env.reset()
    done = False
    truncated = False
    i = 1
    while not done and not truncated:
        action, _ = model.predict(obs)
        obs, reward, truncated, done, info = env.step(action)
        env.render()
        print(''.join([80 * '-', '\nEpisode/ ', str(i), '\nReward/ ', str(reward),
              '\nObservation/ ', str(obs), '\nInfo/ ', str(info), '\n', 80 * '-']))
        i += 1


def run_rbc(env):
    obs, _ = env.reset()
    done = False
    truncated = False
    i = 1
    while not done and not truncated:
        # RBC LOGIC...
        action = env.action_space.sample()
        print('Action = ', str(action))
        obs, reward, truncated, done, info = env.step(action)
        env.render()
        print(''.join([80 * '-', '\nEpisode/ ', str(i), '\nReward/ ', str(reward),
              '\nObservation/ ', str(obs), '\nInfo/ ', str(info), '\n', 80 * '-']))
        i += 1


if __name__ == '__main__':
    env = gym.make('base-v0', control_horizon=CONTROL_HORIZON, check_done_time=CHECK_DONE_TIME,
                   max_deviation=MAX_DEVIATION)
    # train(env)
    # run(env)
    run_rbc(env)
    env.close()
