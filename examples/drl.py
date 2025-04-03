#!/usr/bin/env python3

import melgym
import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make('pressure')

# Training
agent = PPO('MlpPolicy', env, device='cpu')
agent.learn(total_timesteps=5_000, progress_bar=True)

# Evaluation
obs, info = env.reset()
done = trunc = False

while not (done or trunc):
    env.render()
    act, _ = agent.predict(obs)
    obs, rew, done, trunc, info = env.step(act)      

env.close()