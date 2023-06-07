import os
import shutil
import subprocess
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gymnasium import Env, spaces
from melkit.toolkit import Toolkit

from ..utils.constants import *


class EnvHVAC(Env):
    """
    HVAC control environment. Inlet air velocities are adjusted according to HVAC served room states.
    """
    metadata = {'render_modes': ['plot']}

    def __init__(self, n_obs, n_actions, input_file, control_horizon=100, max_deviation=20, max_vel=10, render_mode=None):
        """
        Class constructor.

        Args:
            n_obs (int): number of HVAC served rooms.
            n_actions (int): number of inlet air velocities to be controlled.
            input_file (str): name of the file with the MELGEN/MELCOR input data.
            control_horizon (int, optional): number of simulation cycles. Defaults to 100.
            max_deviation (float, optional): maximum distance allowed from original pressures.
            max_vel (float): maximum value for control actions. Defaults to 10 (m/s).
            render_mode (str): render option.
        """

        # Observation space
        low_obs = np.zeros(n_obs)
        high_obs = np.inf * np.ones(n_obs)
        self.observation_space = spaces.Box(
            low=low_obs, high=high_obs, dtype=np.float64)

        # Action space
        low_act = np.zeros(n_actions)
        high_act = max_vel * np.ones(n_actions)
        self.action_space = spaces.Box(
            low=low_act, high=high_act, dtype=np.float64)

        # Files
        self.input_path = os.path.join(DATA_DIR, input_file)
        self.melin_path = os.path.join(OUTPUT_DIR, 'MELIN')
        self.melog_path = os.path.join(OUTPUT_DIR, 'MELOG')

        # Auxiliar variables
        self.control_horizon = control_horizon
        self.max_deviation = max_deviation
        self.edf_path = None
        self.steps_counter = 0

        # Tookit
        self.toolkit = Toolkit(self.input_path)

        # Render
        if render_mode is not None:
            if render_mode in self.metadata['render_modes']:
                self.render_mode = render_mode
                plt.ion()
            else:
                raise Exception(
                    'The specified render format is not available.')

    def reset(self, seed=None, options=None):
        """
        Environment initialization.
            1. Cleans past simulations outputs.
            2. Generates a copy from original input file.
            3. Updates simulation times according to the specified control horizon.
            4. Executes MELGEN, generating an initial restart file.
            5. Executes MELCOR, creating the EDF file.
            6. Reads the first observation from the last EDF record.

        After that, CFs redefinitions are added to the MELCOR input for future control steps.

        Args:
            seed (int, optional): random seed. Defaults to None.
            options (dict, optional): reset options. Defaults to None.

        Raises:
            Exception: if an EDF file is not found in the output directory.

        Returns:
            np.array: pressures after first simulation.
            dict: last time and pressures record.
        """

        # Env setup
        self.__clean_out_files()
        shutil.copy(self.input_path, self.melin_path)
        self.steps_counter = 1
        self.__update_time()

        # First simulation
        with open(self.melog_path, 'a') as log:
            subprocess.call([MELGEN_PATH, self.melin_path],
                            cwd=OUTPUT_DIR, stdout=log, stderr=subprocess.STDOUT)
            subprocess.call([MELCOR_PATH, self.melin_path],
                            cwd=OUTPUT_DIR, stdout=log, stderr=subprocess.STDOUT)

        # Get initial results from EDF
        edf_files = [os.path.join(OUTPUT_DIR, file) for file in os.listdir(
            OUTPUT_DIR) if file.endswith('.DAT')]
        self.edf_path = edf_files[0] if edf_files else None

        if self.edf_path:
            info = {**{'time':0.0}, **self.__get_last_record()}
            obs = [value for key, value in info.items() if key.startswith('CV')]
        else:
            raise Exception(''.join(['EDF not found in ', OUTPUT_DIR]))

        # Add CFs redefinition to MELCOR input
        self.__add_cfs_redefinition()

        return np.array(obs, dtype=np.float64), info

    def step(self, action):
        """
        Performs a control action by adjusting inlet air velocities.
            1. Modifies those input CFs related to inlet FL velocities.
            2. Updates TEND.
            3. Executes a MELCOR simulation during the number of cycles specified in the control horizon.
            4. Gets the last values recorded in the EDF.
            5. Computes the reward based on pressure requirements.
            6. Evaluates the episode termination condition.

        Args:
            action (np.array): new inlet air velocities.

        Returns:
            np.array: environment observation after control action (current pressures).
            float: reward value. Computed as the sum of the distances of every room pressure to its initial value.
            bool: termination flag. It will be True when pressures converge.
            bool: truncated. Not used.
            dict: additional step information (current cycle and pressures).
        """

        # Input edition
        self.__update_time()
        self.__update_cfs(action)

        # MELCOR simulation
        with open(self.melog_path, 'a') as log:
            subprocess.call([MELCOR_PATH, 'ow=o', 'i=' + self.melin_path],
                            cwd=OUTPUT_DIR, stdout=log, stderr=subprocess.STDOUT)

        # Get results
        info = self.__get_last_record()
        obs = [value for key, value in info.items() if key.startswith('CV')]
        reward, distances = self.__compute_distances(info)
        terminated = self.__check_termination(distances, self.max_deviation)
        truncated = self.__check_truncation()

        return np.array(obs, np.float64), -reward, truncated, terminated, info

    def render(self, time_bt_frames=0.1):
        """
        Plots the pressure values evolution recorded in the EDF.

        Args:
            time_bt_frames (int, optional): time to wait after plotting. Defaults to 0.1.

        Raises:
            Exception: if no step has been performed before render.
        """

        info = self.__get_last_record()

        df = pd.read_csv(self.edf_path, delim_whitespace=True, header=None)
        df = df.set_index(0)
        column_names = [key for key in info.keys()
                        if key.startswith('CV')]
        df.columns = column_names

        df = df.apply(pd.to_numeric, errors='coerce')

        df.plot()

        plt.draw()
        plt.pause(time_bt_frames)
        plt.close('all')

    def close(self):
        """
        Cleans every output file except PTFs (*PTF) and EDFs (*.DAT).
        """
        for file in os.listdir(OUTPUT_DIR):
            if not file.endswith(('PTF', '.DAT')):
                os.remove(os.path.join(OUTPUT_DIR, file))

    def __clean_out_files(self):
        """
        Cleans the output directory where past simulation files are stored.
        """
        for file in os.listdir(OUTPUT_DIR):
            os.remove(os.path.join(OUTPUT_DIR, file))

    def __update_time(self):
        """
        Edits the TEND register in the input file according to the specified control horizon.

        Args:
            melin_path (str): path to the input file to be edited.

        Raises:
            Exception: if no TEND register is found in the input file.
            Exception: if a  step is performed without an initial call to reset.
        """
        if self.steps_counter > 0:
            with open(self.melin_path, 'r+') as f:
                lines = f.readlines()
                edit_line = -1
                for i, line in enumerate(lines):
                    if 'TEND' in line:
                        edit_line = i
                        current_tend = int(
                            line.split()[1]) if self.steps_counter > 1 else 0
                        break
                if edit_line != -1:
                    new_tend = str(
                        current_tend + self.control_horizon * self.steps_counter)
                    lines[edit_line] = ''.join(['TEND ', new_tend, '\n'])

                    # Update file with new TEND
                    f.seek(0)
                    f.writelines([line.strip() + '\n' for line in lines])
                    f.truncate()
                else:
                    raise Exception(
                        ''.join(['TEND not especified in ', self.input_path]))
            self.steps_counter += 1
        else:
            raise Exception('No reset has been done before step')

    def __get_last_record(self):
        """
        Reads the last recorded values in the EDF.

        Returns:
            dict: last registered values (i.e. time and pressures).
        """

        record = {}

        pressure_vars = [
            var for var in self.toolkit.get_edf_vars() if var.startswith('CVH-P.')]

        with open(self.edf_path, 'r') as f:
            last_line = f.readlines()[-1].split()

            record['time'] = float(last_line[0])

            for i, var in enumerate(pressure_vars):
                cv_number = var[var.find('.') + 1:]
                cv_id = ''.join(['CV', '0' * (3 - len(cv_number)), cv_number])
                record[cv_id] = np.float64(last_line[i+1])

        return record

    def __add_cfs_redefinition(self, cf_keyword='CONTROLLER'):
        """
        Includes in the MELCOR input the definitions of the CFs to be overwritten.

        Args:
            cf_keyword (str, optional): keyword used for identifying an air inlet controller. Defaults to 'CONTROLLER'.
        """

        cf_headlines = []

        controllers = [cf for cf in self.toolkit.get_cf_list(
        ) if cf.get_field('CFNAME').startswith(cf_keyword)]

        for cf in controllers:
            cf_headlines.append(str(cf).split('\n')[0])

        with open(self.melin_path, 'r+') as f:
            lines = f.readlines()
            last_line = lines[-1]

            f.seek(0)

            for line in lines[:-1]:
                f.write(line)

            f.write(''.join([30 * '*', ' CONTROLLERS ', 30 * '*', 2 * '\n']))

            for headline in cf_headlines:
                f.write(headline + '\n')

            f.write('\n' + last_line)
            f.truncate()

    def __update_cfs(self, action):
        """
        Reads the input from end to start replacing CF scale factors by action values.

        Args:
            action (np.array): new flow velocities (CF scale factors).
        """
        with open(self.melin_path, 'r+') as f:
            lines = f.readlines()
            i = len(action) - 1

            # Read file, from end to start
            for j in range(len(lines)-1, -1, -1):
                line = lines[j].strip()
                if line.startswith('CF'):
                    fields = line.split()
                    # Overwrite CF scale factor
                    fields[4] = str(action[i])
                    lines[j] = ' '.join(fields) + '\n'
                    i -= 1
                    if i < 0:
                        break
            f.seek(0)
            f.writelines(lines)
            f.truncate()

    def __compute_distances(self, last_record):
        """
        Computes the sum of the distances of each CV pressure to their initial (target) value.

        Args:
            last_record (dict): last recorded data in the EDF.

        Returns:
            float: sum of all the distances between CV pressures to their initial values.
            dict: individual pressure distances.
        """

        distances = {}

        ids = [id for id in last_record if id.startswith('CV')]

        for cv_id in ids:
            distances[cv_id] = abs(
                self.__get_initial_pressure(cv_id) - last_record[cv_id])

        return sum(distances.values()), distances

    def __get_initial_pressure(self, cv_id):
        """
        Returns the initial pressure of a CV.

        Args:
            cv_id (str): CV identifier (CVnnn).

        Returns:
            float: initial pressure.
        """

        cv = self.toolkit.get_cv(cv_id)

        return float(cv.get_field('PVOL'))

    def __check_termination(self, distances, max_deviation=20):
        """
        Checks the completion of an episode. 
        An episode ends when at least one pressure has moved epsilon Pa away from its original value.

        Args:
            distances (dict): computed distances from original to current pressures.
            max_deviation (float, optional): maximum allowed deviation. Defaults to 20.

        Returns:
            bool: True if at least one pressure is out of its allowed limits.
        """
        return any(valor > max_deviation for valor in distances.values())

    def __check_truncation(self, max_steps=100000):
        """
        Checks the truncation condition of an episode. 
        An episode is truncated if the number of steps is greater than a value specified in max_steps.

        Args:
            max_steps (int, optional): maximum number of steps for an episode. Defaults to 100000.

        Returns:
            bool: True if the maximum number of steps have been raised.
        """
        return self.steps_counter > max_steps
