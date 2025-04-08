import melgym
import gymnasium as gym

from stable_baselines3 import PPO

env = gym.make('pressure-v0')

# Training
agent = PPO('MlpPolicy', env)
agent.learn(total_timesteps=10_000)

# Evaluation
obs, _ = env.reset()
done = trunc = False

while not (done or trunc):
    env.render()
    act, _ = agent.predict(obs)
    obs, rew, done, trunc, info = env.step(act)      

env.close()