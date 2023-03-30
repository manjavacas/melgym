import gymnasium as gym
import numpy as np

import os
import subprocess

from gymnasium import spaces
from ..utils import edf

RUN_COMMAND = 'sample'
CLEAN_COMMAND = 'clean'
CLEAN_ALL_COMMAND = 'cleanall'

INPUT_FILENAME = 'sample.inp'

ROOT_DIR = os.getcwd()

DATA_DIR= os.path.join(ROOT_DIR, 'melgym', 'data')
EXEC_DIR = os.path.join(ROOT_DIR, 'melgym', 'exec')

INPUT_PATH = os.path.join(DATA_DIR, INPUT_FILENAME)

MELGEN_PATH = os.path.join(EXEC_DIR, 'melgen-fusion-186_bdba')
MELCOR_PATH = os.path.join(EXEC_DIR, 'melcor-fusion-186_bdba')

EDF_PATH = os.path.join(EXEC_DIR, 'VARIABLES.DAT')


class SampleEnv(gym.Env):
    def __init__(self, obs_size, n_actions):
        '''
        Environment constructor.
        '''
        low_obs = np.zeros(obs_size)
        high_obs = np.inf * np.ones(obs_size)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs)

        low_action = -np.inf * np.ones(n_actions)
        high_action = np.inf * np.ones(n_actions)
        self.action_space = spaces.Box(low=low_action, high=high_action)

    def step(self, action):
        return NotImplementedError

    def reset(self, seed=None, options=None):
        '''
        Clears output files and returns an initial observation.
        '''
        
        subprocess.run(['make', CLEAN_ALL_COMMAND], cwd=EXEC_DIR)
        subprocess.run(['make', RUN_COMMAND], cwd=EXEC_DIR)
        
        obs = edf.make_observation(input_file=INPUT_PATH, edf_file=EDF_PATH)

        return np.array(obs, dtype=np.float32), {}

    def render(self):
        return NotImplementedError

    def close(self):
        subprocess.run(['make', CLEAN_COMMAND], cwd=EXEC_DIR)
