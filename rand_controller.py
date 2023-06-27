#!/usr/bin/python3

import melgym
import gymnasium as gym


MAX_DEVIATION = 20
CHECK_DONE_TIME = 1_000
CONTROL_HORIZON = 5


def summary(episode, action, obs, reward, info):
    print(''.join(['[ACTION] ', str(action), '\n', 80 * '-', '\n[EPISODE] ', str(episode),
                   '\n[REWARD] ', str(reward), '\n[OBSERVATION] ', str(obs),
                   '\n[TIME] ', str(info['time']), '\n[PRESSURES] ', str(
                       info['pressures']),
                   '\n[DISTANCES] ', str(info['distances']), '\n', 80 * '-']))


def rand_control(env):
    obs, _ = env.reset()
    done = False
    truncated = False
    n_episode = 1
    while not done and not truncated:
        # action = env.action_space.sample()
        action = [10.0]
        obs, reward, truncated, done, info = env.step(action)
        summary(n_episode, action, obs, reward, info)
        env.render()
        n_episode += 1


if __name__ == '__main__':
    env = gym.make('branch-v0', control_horizon=CONTROL_HORIZON, check_done_time=CHECK_DONE_TIME,
                   max_deviation=MAX_DEVIATION, render_mode='distances')
    rand_control(env)
    env.close()
