#!/usr/bin/python3

import gymnasium as gym
import melgym

from melgym.utils.constants import N_HVAC_SERVED_ROOMS, N_HVAC_BRANCHES


def run():
    env = gym.make('hvac-v0', n_obs=N_HVAC_SERVED_ROOMS,
                   n_actions=N_HVAC_BRANCHES, control_horizon=100)
    obs, info = env.reset()
    print('observacion = ', obs)
    print('tiempo = ', info)

    env.close()


if __name__ == '__main__':
    run()
