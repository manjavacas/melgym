#!/usr/bin/env python3

import melgym
import gymnasium as gym

from gymnasium.wrappers import RescaleAction, NormalizeObservation, NormalizeReward

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback


def apply_wrappers(env):
    """
    Applies a series of wrappers to the environment.

    Args:
        env (gym.Env): The environment to wrap.
    Returns:
        gym.Env: The wrapped environment.
    """
    env = RescaleAction(env, min_action=-1, max_action=1)
    env = NormalizeObservation(env)
    env = NormalizeReward(env)
    env = Monitor(env, filename='monitor.csv')
    return env


# Training environment
env = apply_wrappers(gym.make('pressure-v0'))

# Evaluation environment
eval_env = apply_wrappers(gym.make('pressure-v0'))

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path='./best_model/',
    log_path='./logs/',
    eval_freq=2000,
    n_eval_episodes=5,
    deterministic=True,
    render=False
)

# Training
agent = PPO('MlpPolicy', env=env, device='cpu', verbose=True)
agent.learn(total_timesteps=10_000, progress_bar=True,
            callback=eval_callback)

# Test best model
agent = PPO.load('best_model/best_model', env=env, device='cpu')
obs, info = env.reset()
done = trunc = False
while not (done or trunc):
    action, _ = agent.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    print(f"Reward: {reward}, Info: {info}")
    env.render()

env.close()
