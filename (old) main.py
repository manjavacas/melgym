#!/usr/bin/python3

import melgym
import copy

from melgym.utils.callbacks import MetricsCallback
from melgym.utils.wrappers import DenormaliseActionsWrapper

import gymnasium as gym

from stable_baselines3 import PPO, DDPG, TD3, SAC
from stable_baselines3.common.callbacks import EvalCallback


MAX_DEVIATION = 20
CHECK_DONE_TIME = 1_000
CONTROL_HORIZON = 5

TRAIN_TIMESTEPS = 300_000
EVAL_FREQ = 10_000
EVAL_EPISODES = 1


def train(env):

    # Evaluation environment
    env_eval = copy.deepcopy(env)

    # Callbacks
    eval_callback = EvalCallback(
        env_eval, best_model_save_path='./best_models/', eval_freq=EVAL_FREQ, n_eval_episodes=EVAL_EPISODES)

    # Action denormalisation [-1, 1] -> [0, 10]
    env = DenormaliseActionsWrapper(env)
    env_eval = DenormaliseActionsWrapper(env_eval)

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
        print(''.join([80 * '-', '\nEpisode/ ', str(i), '\nReward/ ', str(reward),
              '\nObservation/ ', str(obs), '\nInfo/ ', str(info), '\n', 80 * '-']))
        env.render()
        i += 1


def rand_control(env):
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
        # env.render()
        i += 1


if __name__ == '__main__':
    env = gym.make('branch-nl-v0', control_horizon=CONTROL_HORIZON, check_done_time=CHECK_DONE_TIME,
                   max_deviation=MAX_DEVIATION)
    train(env)
    run(env)
    # rand_control(env)
    env.close()
