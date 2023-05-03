from gymnasium import Env, spaces

from melkit.toolkit import Toolkit

import os
import subprocess
import numpy as np

from ..utils import edf, rewards
from ..utils.constants import DATA_DIR, EXEC_DIR, CLEAN_COMMAND, CLEAN_ALL_COMMAND

MAX_AR_GRWTS = 22.2
# MAX_AIR_PERCENT = 10.

RUN_COMMAND = 'sgs'
INPUT_FILENAME = 'sgs.inp'
INPUT_PATH = os.path.join(DATA_DIR, INPUT_FILENAME)
EDF_PATH = os.path.join(EXEC_DIR, 'VARIABLES.DAT')
OUTPUT_FILE = 'melcor_exec.out'

EPISODE_STEPS = 100


class SGSEnv(Env):
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
            low=low_obs, high=high_obs, shape=(obs_size,), dtype=np.float32)

        # An action is a continuous number between 0 and MAX_AR_GRWTS
        self.action_space = spaces.Box(
            low=0, high=MAX_AR_GRWTS, shape=(n_actions,), dtype=np.float32)

        self.toolkit = Toolkit(INPUT_PATH)

        self.n = 0

    def reset(self, seed=None, options=None):
        '''
        Clears output files and returns an initial observation.
        '''

        with open(OUTPUT_FILE, 'a') as output_file:
            subprocess.run(['make', CLEAN_ALL_COMMAND],
                           cwd=EXEC_DIR, stdout=output_file)
            subprocess.run(['make', RUN_COMMAND],
                           cwd=EXEC_DIR, stdout=output_file)

        obs = edf.make_sgs_observation(
            input_file=INPUT_PATH, edf_file=EDF_PATH, verbose=1)

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

        self.toolkit.remove_comments(overwrite=True)

        with open(OUTPUT_FILE, 'a') as output_file:
            subprocess.run(['make', CLEAN_ALL_COMMAND],
                           cwd=EXEC_DIR, stdout=output_file)
            subprocess.run(['make', RUN_COMMAND],
                           cwd=EXEC_DIR, stdout=output_file)

        new_obs = edf.make_sgs_observation(
            input_file=INPUT_PATH, edf_file=EDF_PATH, verbose=1)

        reward = rewards.get_sgs_reward_3(new_obs)

        done = False

        print(
            f'Action = {action}\nReward = {reward}\nNew observation = {new_obs}\nDone = {done}\n')

        return new_obs, reward, done, False, {}

    def render(self):
        return NotImplementedError

    def close(self):
        with open(OUTPUT_FILE, 'a') as output_file:
            subprocess.run(['make', CLEAN_COMMAND],
                           cwd=EXEC_DIR, stdout=output_file)
