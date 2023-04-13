import gymnasium as gym

from gymnasium import spaces

from melkit.toolkit import Toolkit

import os
import subprocess
import numpy as np

from ..utils import edf, rewards


MAX_AR_GRWTS = 44.4
MAX_AIR_PERCENT = 10.

####################### EXEC CONSTANTS #######################

RUN_COMMAND = 'sgs'
CLEAN_COMMAND = 'clean'
CLEAN_ALL_COMMAND = 'cleanall'

INPUT_FILENAME = 'sgs2.inp'

ROOT_DIR = os.getcwd()

DATA_DIR = os.path.join(ROOT_DIR, 'melgym', 'data')
EXEC_DIR = os.path.join(ROOT_DIR, 'melgym', 'exec')

INPUT_PATH = os.path.join(DATA_DIR, INPUT_FILENAME)

MELGEN_PATH = os.path.join(EXEC_DIR, 'melgen-fusion-186_bdba')
MELCOR_PATH = os.path.join(EXEC_DIR, 'melcor-fusion-186_bdba')

EDF_PATH = os.path.join(EXEC_DIR, 'VARIABLES.DAT')

###############################################################


class SGSEnv(gym.Env):
    '''
    SGS environment class
    '''

    def __init__(self, obs_size, n_actions):
        '''
        Environment constructor.
        '''
        low_obs = np.zeros(obs_size)
        high_obs = 100. * np.ones(obs_size)

        # So many observed variables as SGS rooms
        self.observation_space = spaces.Box(
            low=low_obs, high=high_obs, shape=(4,), dtype=np.float32)

        # An action is a continuous number between 0 and MAX_AR_GRWTS
        self.action_space = spaces.Box(
            low=0, high=MAX_AR_GRWTS, shape=(1,), dtype=np.float32)

        self.toolkit = Toolkit(INPUT_PATH)

    def reset(self, seed=None, options=None):
        '''
        Clears output files and returns an initial observation.
        '''

        subprocess.run(['make', CLEAN_ALL_COMMAND], cwd=EXEC_DIR)
        subprocess.run(['make', RUN_COMMAND], cwd=EXEC_DIR)

        obs = edf.make_sgs_observation(
            input_file=INPUT_PATH, edf_file=EDF_PATH)

        return np.array(obs, dtype=np.float32), {}

    def step(self, action):
        '''
        Modifies the ammount of Argon rejected through G-RWTS.
        Also returns a new observation, computes reward and checks termination.
        '''
        cf = self.toolkit.get_cf('CF101')

        # Update G-RWTS CF value
        cf.update_field('ARADCN_0', action[0])
        self.toolkit.update_object(cf, overwrite=True)

        subprocess.run(['make', CLEAN_ALL_COMMAND], cwd=EXEC_DIR)
        subprocess.run(['make', RUN_COMMAND], cwd=EXEC_DIR)

        new_obs = edf.make_sgs_observation(
            input_file=INPUT_PATH, edf_file=EDF_PATH)

        reward = rewards.get_sgs_reward(new_obs, MAX_AIR_PERCENT)

        done = all(value < 10. for value in new_obs)

        return new_obs, reward, done, False, {}

    def render(self):
        return NotImplementedError

    def close(self):
        subprocess.run(['make', CLEAN_COMMAND], cwd=EXEC_DIR)
