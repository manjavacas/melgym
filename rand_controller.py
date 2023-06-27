#!/usr/bin/python3

import melgym
import gymnasium as gym


MAX_DEVIATION = 20
CHECK_DONE_TIME = 1_000
CONTROL_HORIZON = 5

def rand_control(env):
    obs, _ = env.reset()
    done = False
    truncated = False
    i = 1
    while not done and not truncated:
        action = env.action_space.sample()
        # action = [7.785]
        print('Action = ', str(action))
        obs, reward, truncated, done, info = env.step(action)
        print(''.join([80 * '-', '\nEpisode/ ', str(i), '\nReward/ ', str(reward),
              '\nObservation/ ', str(obs), '\nInfo/ ', str(info), '\n', 80 * '-']))
        env.render()
        i += 1


if __name__ == '__main__':
    env = gym.make('branch-nl-v0', control_horizon=CONTROL_HORIZON, check_done_time=CHECK_DONE_TIME,
                   max_deviation=MAX_DEVIATION, render_mode='distances')
    rand_control(env)
    env.close()
