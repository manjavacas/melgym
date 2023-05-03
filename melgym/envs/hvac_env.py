from gymnasium import Env, spaces

from melkit.toolkit import Toolkit

import os
import subprocess
import numpy as np

from ..utils import edf, rewards
from ..utils.constants import DATA_DIR, EXEC_DIR, CLEAN_COMMAND, CLEAN_ALL_COMMAND

AUX_CVS = ['CV001']
SUPPLY_IDS = ['OVERPRESSURE', 'UNDERPRESSURE']

RUN_COMMAND = 'hvac'
INPUT_FILENAME = 'hvac.inp'
INPUT_PATH = os.path.join(DATA_DIR, INPUT_FILENAME)
EDF_PATH = os.path.join(EXEC_DIR, 'VARIABLES.DAT')

ACTIONS = {
    0: (0, 0),
    1: (0.1, 0),
    2: (0, 0.1),
    3: (0.1, 0.1),
    4: (-0.1, 0),
    5: (0, -0.1),
    6: (-0.1, -0.1)
}


class HVACEnv(Env):
    '''
    A sample MELCOR environment class.
    '''

    def __init__(self, obs_size, n_actions):
        '''
        Environment constructor.
        '''
        low_obs = np.zeros(obs_size)
        high_obs = np.inf * np.ones(obs_size)

        # So many observed variables as CVs representing HVAC served rooms
        self.observation_space = spaces.Box(low=low_obs, high=high_obs)

        # So many actions (between 0 and 6) as CVs representing HVAC served rooms
        # Each action is an (overpressure, underpressure) increment/decrement tuple
        self.action_space = spaces.MultiDiscrete([n_actions] * obs_size)

        self.n_cvs = obs_size
        self.toolkit = Toolkit(INPUT_PATH)

    def step(self, actions):
        '''
        This function defines the execution and effect of an action on the environment. It consists of the following steps:
            - Modifies the scalar values by which the supply velocity is multiplied in cases of overpressure and underpressure.
            - Relaunch the simulation.
            - Get a new observation.
            - Calculates the associated reward.
            - Check if all pressures are at an acceptable level. If so, the episode is terminated.
        '''

        # Get HVAC-served CVs
        hvac_served_cvs = [
            cv for cv in self.toolkit.get_cv_list() if cv.get_id() not in AUX_CVS]

        # Get supply CFs
        supply_cfs = [cf for cf in self.toolkit.get_cf_list(
        ) if cf.get_field('CFNAME') in SUPPLY_IDS]

        # Update supply CFs

        # TO-FIX

        # for i, cv in enumerate(hvac_served_cvs):
        #     action = ACTIONS[actions[i]]
        #     fls_connected = self.toolkit.get_fl_connections(cv.get_id())
        #     supply_fls = filter(lambda fl: fl.get_field(
        #         'FLNAME').startswith('SUPPLY'), fls_connected)
        #     for fl in supply_fls:
        #         cfs_connected = self.toolkit.get_connected_cfs(fl.get_id())
        #         overpressure_cfs = [cf for cf in cfs_connected if cf.get_field(
        #             'CFNAME') == 'OVERPRESSURE']
        #         underpressure_cfs = [cf for cf in cfs_connected if cf.get_field(
        #             'CFNAME') == 'UNDERPRESSURE']
        #         overpressure_cfs[0].update_field('CFSCAL', float(
        #             overpressure_cfs[0].get_field('CFSCAL')) + action[0])
        #         underpressure_cfs[0].update_field('CFSCAL', float(
        #             underpressure_cfs[0].get_field('CFSCAL')) + action[1])

        #         # [!] TO-FIX
        #         # self.toolkit.update_object(underpressure_cfs[0])
        #         # self.toolkit.update_object(overpressure_cfs[0])

        # Rerun MELCOR
        subprocess.run(['make', CLEAN_ALL_COMMAND], cwd=EXEC_DIR)
        subprocess.run(['make', RUN_COMMAND], cwd=EXEC_DIR)

        # Get new observation
        new_obs = edf.make_hvac_observation(
            input_file=INPUT_PATH, edf_file=EDF_PATH)

        # Get reward (pressure compliance accomplished)
        reward = rewards.get_hvac_reward(new_obs)

        # Check episode termination
        done = (reward == self.n_cvs)

        return new_obs, reward, done, False, {}

    def reset(self, seed=None, options=None):
        '''
        Clears output files and returns an initial observation.
        '''

        subprocess.run(['make', CLEAN_ALL_COMMAND], cwd=EXEC_DIR)
        subprocess.run(['make', RUN_COMMAND], cwd=EXEC_DIR)

        obs = edf.make_hvac_observation(
            input_file=INPUT_PATH, edf_file=EDF_PATH)

        return np.array(obs, dtype=np.float32), {}

    def render(self):
        return NotImplementedError

    def close(self):
        subprocess.run(['make', CLEAN_COMMAND], cwd=EXEC_DIR)