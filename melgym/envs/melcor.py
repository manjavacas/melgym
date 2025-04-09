"""
Generic MELCOR environment.

Custom environments must inherit from this class.
"""

import os
import shutil
import subprocess

import gymnasium as gym
import numpy as np

from melkit.toolkit import Toolkit
from typing import Optional
from datetime import datetime

from ..utils.constants import OUTPUT_DIR, MELCOR_PATH, MELGEN_PATH
from ..utils.exceptions import MelgymError, MelgymWarning


class MelcorEnv(gym.Env):
    """
    MELCOR control environment.

    This environment redefines CF scale factors to control the variables recorded in the EDF. Between restarts, each CFnnn00 entry (automatically included in the MELCOR input) is rewritten after a specified number of timesteps, as defined by the control horizon.
    """

    def __init__(
        self,
        melcor_model: str,
        control_cfs: list[str],
        min_action_value: float,
        max_action_value: float,
        control_horizon: int = 10,
        output_dir: Optional[str] = None,
        melgen_path: Optional[str] = None,
        melcor_path: Optional[str] = None
    ):
        """
        Initializes the MELCOR environment.

        Args:   
            melcor_model (str): Path to the MELCOR model file.
            control_cfs (list[str]): List of controlled CFs.
            min_action_value (float): Minimum action value.
            max_action_value (float): Maximum action value.
            control_horizon (int): Control horizon (timesteps between actions).
            output_dir (Optional[str]): Directory name for output files. If None, a default directory is used.
            render_mode (Optional[str]): Mode for rendering the environment.
            melgen_path (Optional[str]): Path to the MELGEN executable. If None, the default path in exec directory is used.
            melcor_path (Optional[str]): Path to the MELCOR executable. If None, the default path in exec directory is used.
        """

        # Files and paths
        self.melcor_model = melcor_model
        model_name = os.path.splitext(os.path.basename(melcor_model))[0]

        if output_dir is None:
            self.output_dir = os.path.join(
                OUTPUT_DIR, model_name + f'_{datetime.now().strftime("%Y_%m_%d-%H_%M_%S")}')
        else:
            self.output_dir = os.path.join(OUTPUT_DIR, output_dir)

        self.melin_path = os.path.join(self.output_dir, 'MELIN')
        self.melog_path = os.path.join(self.output_dir, 'MELOG')
        self.edf_path = os.path.join(self.output_dir, 'MELEDF')

        self.melgen_path = melgen_path if melgen_path is not None else MELGEN_PATH
        self.melcor_path = melcor_path if melcor_path is not None else MELCOR_PATH

        if not os.path.isfile(self.melgen_path):
            raise FileNotFoundError(
                f"MELGEN executable not found at {self.melgen_path}")

        if not os.path.isfile(self.melcor_path):
            raise FileNotFoundError(
                f"MELCOR executable not found at {self.melcor_path}")

        self.toolkit = None

        # Observation and action spaces
        self.control_cfs = control_cfs
        self.controlled_values = Toolkit(self.melcor_model).get_edf_vars()
        if 'TIME' in self.controlled_values:
            self.controlled_values.remove('TIME')

        self.action_space = gym.spaces.Box(
            low=min_action_value,
            high=max_action_value,
            shape=(len(control_cfs),),
            dtype=np.float16
        )

        n_obs = len(self.controlled_values)
        self.observation_space = gym.spaces.Box(
            low=-np.inf * np.ones(n_obs),
            high=np.inf * np.ones(n_obs),
            dtype=np.float64
        )

        # Simulation parameters
        self.control_horizon = control_horizon
        self.n_steps = 0
        self.current_tend = 0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Resets the environment to an initial state and returns the first observation.

        Args:
            seed (Optional[int]): Seed for random number generation.
            options (Optional[dict]): Additional options for resetting. Not used by default.
        Returns:
            tuple: A tuple containing the initial observation and info.
        Raises:
            Exception: If the MELGEN execution fails.
        """
        super().reset(seed=seed)

        if os.path.exists(self.output_dir):
            # Clean output directory from previous runs
            self._clean_out_files()
        else:
            # Create output folder and copy input file
            os.makedirs(self.output_dir, exist_ok=True)

        shutil.copy(self.melcor_model, self.melin_path)

        self.toolkit = Toolkit(self.melin_path)
        self.toolkit.remove_comments(overwrite=True)

        # Set initial TEND
        self.n_steps = 1
        self._update_time()

        # MELGEN execution
        try:
            with open(self.melog_path, 'a') as log:
                subprocess.run([self.melgen_path, self.melin_path],
                               cwd=self.output_dir, stdout=log, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            raise MelgymError(f"MELGEN execution failed: {e}")

        # Initial state
        info = {'time': 0.0}
        obs = np.zeros(self.observation_space.shape,
                       dtype=self.observation_space.dtype)

        # Add CFs redefinition to MELCOR input
        self._add_cfs_redefinition()

        return obs, info

    def step(self, action):
        """
        Executes a step in the MELCOR environment by applying the given action, running a MELCOR simulation during a given control horizon, and retrieving the latest state.

        Args:
            action (np.array): An array of control values (CFs scale factors) to be applied.

        Returns:
            tuple:
                - np.array: Observations, representing the latest EDF values after simulation.
                - float: Reward calculated based on the latest state.
                - bool: Whether the episode has terminated.
                - bool: Whether the episode was truncated.
                - dict: Additional metadata, including:
                    - "TIME" (float): The current simulation time.
                    - Controlled variable names as keys with their respective values.

        Raises:
            Exception: If reset() has not been called before step().
            Exception: If the MELCOR execution fails.
        """

        # Update time
        if self.n_steps > 0:
            self._update_time()
            self.n_steps += 1
        else:
            raise MelgymError(
                "Error: reset() has not been called before step()")

        # Apply action
        self._update_cfs(action)

        # MELCOR simulation
        try:
            with open(self.melog_path, 'a') as log:
                subprocess.call([self.melcor_path, 'ow=o', 'i=' + self.melin_path],
                                cwd=self.output_dir, stdout=log, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            raise MelgymError(f"MELCOR execution failed: {e}")

        # Get observation
        time, *obs = self._get_last_edf_data()

        info = {'TIME': time, 'action': action}
        info.update(dict(zip(self.controlled_values, obs)))

        # Check termination / truncation
        termination = self._check_termination(obs, info)
        truncation = self._check_truncation(obs, info)

        info['termination'] = termination
        info['truncation'] = truncation

        # Compute reward
        reward = self._compute_reward(obs, info)

        return np.array(obs, dtype=np.float64), reward, termination, truncation, info

    def render(self):
        """
        Renders the environment. 
        This method should be overridden in subclasses to define the render method.
        """
        raise NotImplementedError

    def close(self):
        """
        Closes the environment and cleans up resources.

        Raises:
            Exception: If the MELCOR or MELGEN processes cannot be terminated.
        """
        try:
            subprocess.run(["pkill", "-f", self.melcor_path], check=False)
            subprocess.run(["pkill", "-f", self.melgen_path], check=False)
        except MelgymWarning as e:
            print(f"Warning: Failed to terminate MELCOR/MELGEN processes: {e}")

    def _clean_out_files(self):
        """
        Cleans the output directory where past simulation files are stored.
        """
        for file in os.listdir(self.output_dir):
            os.remove(os.path.join(self.output_dir, file))

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

    def _add_cfs_redefinition(self):
        """
        Includes the definitions of the CFs to be overwritten in the MELCOR input, inserting them after "*EOR* MELCOR".
        If the marker is not found, the block is inserted before the last line.

        Raises:
            FileNotFoundError: If the input file is not found.
        """
        # Get the headlines of the controlled CFs
        cf_headlines = [str(cf).split('\n')[0]
                        for cf in self.toolkit.get_cf_list()
                        if cf.get_id() in self.control_cfs]

        try:
            with open(self.melin_path, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            raise FileNotFoundError(f"Input file {self.melin_path} not found.")

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

    def _update_cfs(self, action):
        """
        Updates the scale factor of every controlled CF in the MELCOR input according to a given action.

        Args:
            action (np.array): New scale factors to assign to the CFs.

        Raises:
            Exception: If the marker "*EOR* MELCOR" is not found.
            FileNotFoundError: If the input file is not found.
        """
        try:
            with open(self.melin_path, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            raise FileNotFoundError(f"Input file {self.melin_path} not found.")

        # Search for "*EOR* MELCOR" line
        try:
            marker_index = next(i for i, line in enumerate(
                lines) if "*EOR* MELCOR" in line)
        except StopIteration:
            raise MelgymError(
                "Marker '*EOR* MELCOR' not found in the input file.")

        action_idx = 0
        num_actions = len(action)

        # Update scale factors in CFs after marker
        for i in range(marker_index + 1, len(lines)):
            tokens = lines[i].split()
            if not tokens or len(tokens) < 5 or not tokens[0].endswith('00'):
                continue

            cf_id = tokens[0][:-2]

            if cf_id in self.control_cfs and action_idx < num_actions:
                tokens[4] = str(action[action_idx])
                action_idx += 1
                lines[i] = ' '.join(tokens) + '\n'

        with open(self.melin_path, 'w') as f:
            f.writelines(lines)

    def _get_last_edf_data(self):
        """
        Reads the last recorded values from the EDF file.

        Returns:
            np.array: An array containing the last recorded values as np.float64.

        Raises:
            FileNotFoundError: If the EDF file is not found.
            ValueError: If the file cannot be parsed correctly.
        """
        try:
            with open(self.edf_path, 'rb') as edf:
                edf.seek(0, os.SEEK_END)
                file_size = edf.tell()
                buffer_size = min(4096, file_size)
                edf.seek(-buffer_size, os.SEEK_END)

                # Leer los últimos valores
                raw_data = edf.read().decode(errors='ignore').split()
                values = raw_data[-(len(self.controlled_values) + 1):]

                # Convertir a NumPy array con dtype float64
                return np.fromstring(" ".join(values), sep=" ", dtype=np.float64)

        except FileNotFoundError:
            raise FileNotFoundError(f"EDF file {self.edf_path} not found.")
        except ValueError:
            raise ValueError(
                f"Failed to parse numerical values from EDF file: {self.edf_path}")

    def _compute_reward(self, obs, info):
        """
        Computes the reward based on the current state.
        This method should be overridden in subclasses to define the reward function.

        Args:
            obs (np.array): Current observation.
            info (dict): Additional information about the current state.
        """
        raise NotImplementedError

    def _check_termination(self, obs, info):
        """
        Checks if the episode has terminated.
        This method should be overridden in subclasses to define the termination condition.

        Args:
            obs (np.array): Current observation.
            info (dict): Additional information about the current state.
        """
        raise NotImplementedError

    def _check_truncation(self, obs, info):
        """
        Checks if the episode should be truncated.
        This method should be overridden in subclasses to define the truncation condition.

        Args:
            obs (np.array): Current observation.
            info (dict): Additional information about the current state.
        """
        raise NotImplementedError
