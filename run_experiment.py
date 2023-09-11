#!/usr/bin/env python

import argparse
import json
import copy

import numpy as np
import gymnasium as gym

from gymnasium.wrappers.normalize import NormalizeObservation, NormalizeReward

from melgym.utils.callbacks import TbMetricsCallback, EpisodicDataCallback
from melgym.utils.aux import summary

from stable_baselines3 import PPO, DDPG, TD3, SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.noise import NormalActionNoise

ALGORITHMS = {
    'PPO': PPO,
    'DDPG': DDPG,
    'TD3': TD3,
    'SAC': SAC
}


def get_config():
    """
    Parses the experiment json configuration file.

    Returns:
        dict: experiment configuration.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--configuration',
        '-conf',
        required=True,
        type=str,
        dest='configuration',
        help='Path to experiment configuration (JSON file)'
    )
    args = parser.parse_args()

    with open(args.configuration) as json_config:
        config = json.load(json_config)
    return config


def apply_wrappers(env, config):
    """
    Applies the indicated wrappers to the environment.

    Args:
        env (gym.Env): default environment.
        config (dict): configuration dictionary.

    Returns:
        gym.Env: wrapped environment.
    """
    if 'norm_obs' in config['wrappers']:
        env = NormalizeObservation(env)
    if 'norm_rew' in config['wrappers']:
        env = NormalizeReward(env)
    return env


def get_callbacks(env, config):
    """
    Gets the list of callbacks to be applied.

    Args:
        env (gym.Env): default environment.
        config (dict): configuration file info.

    Returns:
        list: list of callbacks to be applied.
    """

    experiment_id = config['id']
    train_config = config['algorithm']['train_params']

    callbacks = []
    if 'EvalCallback' in config['callbacks']:
        # Evaluation environment
        env_eval = copy.deepcopy(env)
        eval_freq = train_config['eval_freq']
        n_eval_episodes = train_config['n_eval_episodes']
        callbacks.append(EvalCallback(
            env_eval, best_model_save_path=config['paths']['best_models_dir'] +
            experiment_id + '/', eval_freq=eval_freq, n_eval_episodes=n_eval_episodes, deterministic=True))
    if 'TbMetricsCallback' in config['callbacks']:
        callbacks.append(TbMetricsCallback())
    if 'EpisodicDataCallback' in config['callbacks']:
        callbacks.append(EpisodicDataCallback(
            save_path=config['paths']['ep_metrics_dir']))

    return callbacks


def train(env, config):
    """
    Model training based on user configuration.

    Args:
        env (gym.Env): training environment
        config (dict): configuration dictionary.
    Raises:
        Exception: if the specified model is not available.
    """

    experiment_id = config['id']
    model_config = config['algorithm']['params']
    total_timesteps = config['algorithm']['train_params']['total_timesteps']

    # Apply specified wrappers
    env = apply_wrappers(env, config)

    # Callbacks
    callbacks = get_callbacks(env, config)

    # Model configuration
    if config['algorithm']['name'] in ALGORITHMS:
        model_class = ALGORITHMS[config['algorithm']['name']]
        model = model_class('MlpPolicy', env, verbose=1,
                            tensorboard_log=config['paths']['tensorboard_dir'] + experiment_id, **model_config)
        # Uncomment for noisy actions
        # model = model_class('MlpPolicy', env, verbose=1,
        #                     tensorboard_log=config['paths']['tensorboard_dir'] + experiment_id, **model_config, action_noise=NormalActionNoise(mean=np.array([0]), sigma=np.array([0.1])))
    else:
        raise Exception('Incorrect algorithm name in configuration file.')

    model.learn(total_timesteps=total_timesteps,
                progress_bar=True, callback=callbacks)

    model.save(config['paths']['best_models_dir'] +
               config['id'] + '/last_model')


def test(env, config):
    """
    Runs a trained model during an episode.

    Args:
        env (gym.Env): environment.
        model_id (str, optional): name of the model to be loaded. Defaults to 'best_model'.
    """
    model_class = ALGORITHMS[config['algorithm']['name']]
    model = model_class.load(
        config['paths']['best_models_dir'] + config['id'] + '/best_model')
    obs, _ = env.reset()
    done = False
    truncated = False
    mean_ep_reward = 0
    n_steps = 1
    while not done and not truncated:
        env.render()
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, truncated, done, info = env.step(action)
        summary(n_steps, action, obs, reward, info)
        mean_ep_reward += reward / n_steps
        n_steps += 1
    print('Mean episode reward = ' + str(mean_ep_reward))


config = get_config()

env = gym.make(config['env']['name'], **config['env']
               ['params'], env_id=config['id'])

# Train / test
if 'train' in config['tasks']:
    train(env, config)
if 'test' in config['tasks']:
    test(env, config)

env.close()
