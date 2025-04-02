
import os
import shutil
import subprocess

import gymnasium as gym
import numpy as np

from melkit.toolkit import Toolkit
from typing import Optional
from datetime import datetime

from ..utils.constants import *


class MelcorEnv(gym.Env):

    def __init__(
        self,
        melcor_model: str,
        control_cfs: list[str],
        min_action_value: float,
        max_action_value: float,
        control_horizon: int = 10,
        render_mode: Optional[str] = None
    ):
        """
        Initializes the MELCOR environment.

        Args:   
            melcor_model (str): Path to the MELCOR model file.
            control_cfs (list[str]): List of controlled CFs.
            min_action_value (float): Minimum action value.
            max_action_value (float): Maximum action value.
            control_horizon (int): Control horizon (timesteps).
            render_mode (Optional[str]): Mode for rendering the environment.
        """

        self.render_mode = render_mode

        ########################## Files #########################

        self.melcor_model = melcor_model

        model_name = os.path.basename(melcor_model)
        self.output_dir = os.path.join(OUTPUT_DIR, model_name.split(
            '.')[0] + f'_{datetime.now().strftime("%Y_%m_%d-%H_%M_%S")}')

        self.melin_path = os.path.join(self.output_dir, 'MELIN')
        self.melog_path = os.path.join(self.output_dir, 'MELOG')

        self.toolkit = None

        ############## Observation and action spaces #############

        self.control_cfs = control_cfs
        self.controlled_values = self._get_controlled_variables(melcor_model)
        if 'TIME' in self.controlled_values:
            self.controlled_values.remove('TIME')

        self.action_space = gym.spaces.Box(
            low=min_action_value,
            high=max_action_value,
            shape=(len(control_cfs),),
            dtype=np.float64
        )

        n_obs = len(self.controlled_values)
        self.observation_space = gym.spaces.Box(
            low=-np.inf * np.ones(n_obs),
            high=np.inf * np.ones(n_obs),
            dtype=np.float64
        )

        #################### Simulation params ###################

        self.control_horizon = control_horizon
        self.n_steps = 0
        self.current_tend = 0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Resets the environment to an initial state and returns the first observation.

        Args:
            seed (Optional[int]): Seed for random number generation.
            options (Optional[dict]): Additional options for resetting.
        Returns:
            tuple: A tuple containing the initial observation and info.
        """
        super().reset(seed=seed)

        # Create output folder and copy input file
        os.makedirs(self.output_dir)
        shutil.copy(self.melcor_model, self.melin_path)

        self.toolkit = Toolkit(self.melin_path)
        self.toolkit.remove_comments(overwrite=True)

        self.n_steps = 1
        self._update_time()

        # MELGEN execution
        with open(self.melog_path, 'a') as log:
            subprocess.call([MELGEN_PATH, self.melin_path],
                            cwd=self.output_dir, stdout=log, stderr=subprocess.STDOUT)

        info = {'time': 0.0}
        obs = np.zeros(self.observation_space.shape,
                       dtype=self.observation_space.dtype)

        # Add CFs redefinition to MELCOR input
        self.__add_cfs_redefinition()

        return obs, info

    def _get_controlled_variables(self, melcor_model):
        """
        Retrieves the variables registered in the EDF and returns a list of their names.

        Args:
            melcor_model (str): Path to the MELCOR model file.
        Returns:
            list[str]: A list of controlled variable names.
        """
        return Toolkit(melcor_model).get_edf_vars()

    def _update_time(self):
        """
        Updates the TEND value in the input file based on the specified control horizon.

        Raises:
            ValueError: If no TEND register is found in the input file.
            FileNotFoundError: If the input file is not found.
        """

        new_tend = str(self.control_horizon * self.n_steps)

        try:
            with open(self.melin_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for i, line in enumerate(lines):
                if 'TEND' in line:
                    lines[i] = f"TEND {new_tend}\n"
                    self.current_tend = int(new_tend)
                    break
            else:
                raise ValueError(f"TEND not specified in {self.melin_path}")

            with open(self.melin_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)

        except FileNotFoundError:
            raise FileNotFoundError(f"Input file {self.melin_path} not found.")

    def __add_cfs_redefinition(self):
        """
        Includes the definitions of the CFs to be overwritten in the MELCOR input, inserting them after "*EOR* MELCOR".
        If the marker is not found, the block is inserted before the last line.
        """
        # Get the headlines of the controlled CFs
        cf_headlines = [str(cf).split('\n')[0]
                        for cf in self.toolkit.get_cf_list()
                        if cf.get_id() in self.control_cfs]

        with open(self.melin_path, 'r') as f:
            lines = f.readlines()

        # Search for insertion line
        insert_index = None
        for i, line in enumerate(lines):
            if "*EOR* MELCOR" in line:
                insert_index = i + 1
                break
        if insert_index is None:
            insert_index = len(lines) - 1

        # Add CFs redefinition
        block = f"\n{'*' * 30} CONTROLLERS {'*' * 30}\n" + \
            '\n'.join(cf_headlines) + '\n' + f"{'*' * 73}\n"
        new_lines = lines[:insert_index] + [block] + lines[insert_index:]

        with open(self.melin_path, 'w') as f:
            f.write(''.join(new_lines))
