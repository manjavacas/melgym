#!/usr/bin/python3

import argparse
import json
import copy

import gymnasium as gym

from gymnasium.wrappers.normalize import NormalizeObservation, NormalizeReward

from melgym.utils.callbacks import MetricsCallback
from melgym.utils.aux import summary

from stable_baselines3 import PPO, DDPG, TD3, SAC
from stable_baselines3.common.callbacks import EvalCallback
# from stable_baselines3.common.evaluation import evaluate_policy

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


def train(env, config):
    """
    Model training based on user configuration. Includes:
        - Periodic evaluation.
        - Action denormalisation.
        - Metrics callback.

    Args:
        env (gym.Env): training environment
        config (dict): configuration dictionary.
    Raises:
        Exception: if the specified model is not available.
    """

    experiment_id = config['id']
    model_config = config['algorithm']['params']
    train_config = config['algorithm']['train_params']

    total_timesteps = train_config['total_timesteps']
    eval_freq = train_config['eval_freq']
    n_eval_episodes = train_config['n_eval_episodes']

    # Evaluation environment
    env_eval = copy.deepcopy(env)

    # Callbacks
    eval_callback = EvalCallback(
        env_eval, best_model_save_path='./best_models/' + experiment_id + '/', eval_freq=eval_freq, n_eval_episodes=n_eval_episodes, deterministic=True)

    # Model configuration
    if config['algorithm']['name'] in ALGORITHMS:
        model_class = ALGORITHMS[config['algorithm']['name']]
        model = model_class('MlpPolicy', env, verbose=1,
                            tensorboard_log='./tensorboard/' + experiment_id, **model_config)
    else:
        raise Exception('Incorrect algorithm name in configuration file.')

    model.learn(total_timesteps=total_timesteps,
                progress_bar=True, callback=[eval_callback, MetricsCallback()])


def test(env, config):
    """
    Runs a trained model.

    Args:
        env (gym.Env): environment.
        model_id (str, optional): name of the model to be loaded. Defaults to 'best_model'.
    """
    model_class = ALGORITHMS[config['algorithm']['name']]
    model = model_class.load('./best_models/' + config['id'] + '/best_model')
    obs, _ = env.reset()
    done = False
    truncated = False
    i = 1
    while not done and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, truncated, done, info = env.step(action)
        summary(i, action, obs, reward, info)
        env.render()
        i += 1

    # Optional SB3 evaluation
    # mean_reward, std_reward = evaluate_policy(
    #     model, env, n_eval_episodes=10)
    # print('\n***** SB3 EVALUATION *****\n- Mean reward = ' +
    #       str(mean_reward) + '\n- Std. reward = ' + str(std_reward))


config = get_config()
env = gym.make(config['env']['name'], **config['env']
               ['params'])

# Normalization wrappers
env = NormalizeReward(NormalizeObservation(env))

if 'train' in config['tasks']:
    train(env, config)
if 'test' in config['tasks']:
    test(env, config)

env.close()
