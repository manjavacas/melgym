#!/usr/bin/env python3

import melgym
import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make('pressure')

# Training
agent = PPO('MlpPolicy', env)
agent.learn(total_timesteps=5_000)

# Evaluation
obs, info = env.reset()
done = trunc = False

while not (done or trunc):
    env.render()
    act, _ = agent.predict(obs)
    obs, rew, trunc, done, info = env.step(act)      

env.close()