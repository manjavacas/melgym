#!/usr/bin/env python3

import melgym
import gymnasium as gym
from stable_baselines3 import TD3

env = gym.make('branch_1', render_mode='pressures')

# Training
agent = TD3('MlpPolicy', env)
agent.learn(total_timesteps=10_000)

# Evaluation
obs, info = env.reset()
done = trunc = False

while not (done or trunc):
    env.render()
    act, _ = agent.predict(obs)
    obs, rew, trunc, done, info = env.step(act)      

env.close()