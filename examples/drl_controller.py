#!/usr/bin/env python3

import melgym
import gymnasium as gym
from gymnasium.wrappers import RescaleAction, NormalizeObservation, NormalizeReward

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback

class RenderEveryNSteps(BaseCallback):
    def __init__(self, env, n_steps, verbose=0):
        super().__init__(verbose)
        self.env = env
        self.n_steps = n_steps
        self.last_render = 0

    def _on_step(self) -> bool:
        if (self.num_timesteps - self.last_render) >= self.n_steps:
            self.last_render = self.num_timesteps
            self.render_episode()
        return True

    def render_episode(self):
        obs, info = self.env.reset()
        done = False
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = self.env.step(action)
            self.env.render()


env = gym.make('pressure', render_mode='human')

env = RescaleAction(env, min_action=-1, max_action=1)
env = NormalizeObservation(env)
env = NormalizeReward(env)

render_callback = RenderEveryNSteps(env, n_steps=1_000)

# Training
agent = PPO('MlpPolicy', env=env, device='cpu', verbose=True)
agent.learn(total_timesteps=10_000, progress_bar=True, callback=render_callback)
agent.save('rl_model')

# Evaluation
agent = PPO.load('ppo_pressure', env=env, device='cpu')
obs, info = env.reset()
done = trunc = False

while not (done or trunc):
    action, _ = agent.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    env.render()

env.close()
