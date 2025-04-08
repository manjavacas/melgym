#!/usr/bin/env python3

import melgym
import gymnasium as gym

from datetime import datetime

from gymnasium.wrappers import RescaleAction, NormalizeObservation

from sb3_contrib import RecurrentPPO

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback


RENDER_FREQ = 1
TRAIN_EPISODES = 10_000
EVAL_FREQ = 2_000
EVAL_EPISODES = 5

TRAIN_PARAMS = {
    'ent_coef': 0.01,
    'n_steps': 100,
    'batch_size': 100
}

run_id = 'run_' + datetime.now().strftime("%Y%m%d_%H%M%S")


class RenderCallback(BaseCallback):
    def _on_step(self) -> bool:
        if self.num_timesteps % RENDER_FREQ == 0:
            env.render()
        return True


def apply_wrappers(env):
    """
    Applies a series of Gymnasium wrappers to the environment.

    Args:
        env (gym.Env): The environment to wrap.
    Returns:
        gym.Env: The wrapped environment.
    """
    env = RescaleAction(env, min_action=-1, max_action=1)
    env = NormalizeObservation(env)
    return env


# Training environment
env = apply_wrappers(gym.make('pressure-v0', render_mode='human'))
env = Monitor(env, filename=run_id + '_monitor.csv')

# Evaluation environment
eval_env = apply_wrappers(gym.make('pressure-v0'))
eval_env = Monitor(env, filename=run_id + '_eval_monitor.csv')

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path='best_models/' + run_id,
    eval_freq=EVAL_FREQ,
    n_eval_episodes=EVAL_EPISODES,
    deterministic=True,
    render=False
)

# Training
agent = RecurrentPPO('MlpLstmPolicy', env=env, device='cpu',
                     verbose=True, **TRAIN_PARAMS)
agent.learn(total_timesteps=TRAIN_EPISODES, progress_bar=True,
            callback=[RenderCallback(), eval_callback])

# Test best model
agent = RecurrentPPO.load('best_models/' + run_id +
                          '/best_model', env=env, device='cpu')

obs, info = env.reset()
done = trunc = False

while not (done or trunc):
    action, _ = agent.predict(obs, deterministic=True)
    obs, reward, done, trunc, info = env.step(action)
    print(f"Reward: {reward}, Info: {info}")
    env.render()

env.close()
