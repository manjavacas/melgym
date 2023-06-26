#!/usr/bin/python3

import melgym

import gymnasium as gym
import numpy as np

import copy

from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback


MAX_DEVIATION = 20
CHECK_DONE_TIME = 1_000
CONTROL_HORIZON = 5

TRAIN_TIMESTEPS = 300_000
EVAL_FREQ = 10_000
EVAL_EPISODES = 1


class MetricsCallback(BaseCallback):
    """
    Custom callback for gathering additional simulation data.
    """

    def __init__(self, verbose=0, metrics_folder='metrics', log_freq=1):
        super().__init__(verbose)
        # import os
        # self.metrics_folder = metrics_folder
        # self.log_freq = log_freq
        # if not os.path.exists(self.metrics_folder):
        #     os.makedirs(self.metrics_folder)

    def _on_step(self) -> bool:

        # Tensorboard
        action = self.locals['actions'][-1][-1]
        self.logger.record('sim_data/last_action', action)

        # .CSV
        # episode = self.training_env.get_attr('n_episodes')[-1]
        # if episode == 1 or episode % self.log_freq == 0:
        #     with open(self.metrics_folder + '/episode_' + str(episode) + '.csv', 'a') as f:
        #         f.write(str(action) + '\n')

        return True


def train(env):

    # Evaluation environment
    env_eval = copy.deepcopy(env)

    # Callbacks
    eval_callback = EvalCallback(
        env_eval, best_model_save_path='./best_models/', eval_freq=EVAL_FREQ, n_eval_episodes=EVAL_EPISODES)

    model = DDPG('MlpPolicy', env, verbose=1,
                tensorboard_log='./tensorboard/')
    model.learn(total_timesteps=TRAIN_TIMESTEPS,
                progress_bar=True, callback=[eval_callback, MetricsCallback()])


def run(env, model_id='best_model'):
    model = DDPG.load('./best_models/' + model_id)
    obs, _ = env.reset()
    done = False
    truncated = False
    i = 1
    while not done and not truncated:
        action, _ = model.predict(obs)
        obs, reward, truncated, done, info = env.step(action)
        print(''.join([80 * '-', '\nEpisode/ ', str(i), '\nReward/ ', str(reward),
              '\nObservation/ ', str(obs), '\nInfo/ ', str(info), '\n', 80 * '-']))
        env.render()
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
        print(''.join([80 * '-', '\nEpisode/ ', str(i), '\nReward/ ', str(reward),
              '\nObservation/ ', str(obs), '\nInfo/ ', str(info), '\n', 80 * '-']))
        env.render()
        i += 1


if __name__ == '__main__':
    env = gym.make('branch-nl-v0', control_horizon=CONTROL_HORIZON, check_done_time=CHECK_DONE_TIME,
                   max_deviation=MAX_DEVIATION)
    train(env)
    run(env)
    # run_rbc(env)
    env.close()
