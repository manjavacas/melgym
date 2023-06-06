import os
import shutil
import subprocess
import numpy as np

from gymnasium import Env, spaces
from melkit.toolkit import Toolkit

from ..utils.constants import *


class EnvHVAC(Env):
    """
    HVAC control environment. Inlet air velocities are adjusted according to HVAC served room states.
    """

    def __init__(self, n_obs, n_actions, input_file, control_horizon=100, max_vel=10):
        """
        Class constructor.

        Args:
            n_obs (int): number of HVAC served rooms.
            n_actions (int): number of inlet air velocities to be controlled.
            input_file (str): name of the file with the MELGEN/MELCOR input data.
            control_horizon (int, optional): number of simulation cycles. Defaults to 100.
            max_vel (float): maximum value for control actions. Defaults to 10 (m/s).
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
        self.melcor_log_path = os.path.join(OUTPUT_DIR, 'MELOG')

        # Auxiliar variables
        self.control_horizon = control_horizon
        self.edf_path = None
        self.steps_counter = 0

        # Tookit
        self.toolkit = Toolkit(self.input_path)

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
            np.array: first environment observation.
            dict: auxiliar information (e.g. current cycle).
        """

        # Env setup
        self.__clean_out_files()
        shutil.copy(self.input_path, self.melin_path)
        self.steps_counter += 1
        self.__update_tend()

        # First simulation
        with open(self.melcor_log_path, 'a') as log:
            subprocess.call([MELGEN_PATH, self.melin_path],
                            cwd=OUTPUT_DIR, stdout=log, stderr=subprocess.STDOUT)
            subprocess.call([MELCOR_PATH, self.melin_path],
                            cwd=OUTPUT_DIR, stdout=log, stderr=subprocess.STDOUT)

        # Get initial results from EDF
        edf_files = [os.path.join(OUTPUT_DIR, file) for file in os.listdir(
            OUTPUT_DIR) if file.endswith('.DAT')]
        self.edf_path = edf_files[0] if edf_files else None

        if self.edf_path:
            obs, cycle = self.__get_edf_obs()
        else:
            raise Exception(''.join(['EDF not found in ', OUTPUT_DIR]))

        # Add CFs redefinitions to MELCOR input
        self.__add_cfs_redefinitions()

        return obs, {'cycle': cycle}

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
            np.array: environment observation after control action.
            float: reward value. Computed as the sum of the distances of every room pressure to its initial value.
            bool: termination flag.
            bool: truncated. Not used.
            dict: additional step information (e.g. current cycle and pressures).
        """

        # Input edition
        self.__update_tend()
        self.__update_cfs(action)

        # MELCOR simulation
        with open(self.melcor_log_path, 'a') as log:
            subprocess.call([MELCOR_PATH, 'ow=o', 'i=' + self.melin_path],
                            cwd=OUTPUT_DIR, stdout=log, stderr=subprocess.STDOUT)

        # Get results
        obs, cycle = self.__get_edf_obs()
        reward, pressures_info = self.__compute_distances(obs)
        terminated = self.__check_termination(obs)

        info = {**{'cycle': cycle}, **pressures_info}

        return obs, reward, False, terminated, info

    def render(self):
        raise NotImplementedError

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

    def __update_tend(self):
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

    def __get_edf_obs(self):
        """
        Get the last recorded values in the simulation EDF.

        Returns:
            np.array: last registered values.
            float: simulation cycle when the last EDF values were recorded.
        """
        with open(self.edf_path, 'r') as f:
            last_record = f.readlines()[-1].split()
            last_values = np.array(last_record[1:], dtype=np.float64)
        return last_values, float(last_record[0])

    def __add_cfs_redefinitions(self, cf_keyword='CONTROLLER'):
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

            # Read loop, from end to start
            for j in range(len(lines)-1, -1, -1):
                line = lines[j].strip()
                if line.startswith('CF'):
                    fields = line.split()
                    # Overwrite scale factor
                    fields[4] = str(action[i])
                    lines[j] = ' '.join(fields) + '\n'
                    i -= 1
                    if i < 0:
                        break
            f.seek(0)
            f.writelines(lines)
            f.truncate()

    def __compute_distances(self, obs):
        """
        Computes the sum of the distances of each CV pressure to their original/target values.

        Args:
            obs (np.array): current pressures, obtained from EDF latest record.

        Returns:
            float: sum of the distances between CV pressures and their original values.
            dict: last individual pressures recorded in the EDF.
        """

        distances = []
        current_pressures = {}

        pressure_vars = [
            var for var in self.toolkit.get_edf_vars() if var.startswith('CVH-P.')]

        for i, var in enumerate(pressure_vars):
            cv_number = var[var.find('.') + 1:]
            cv_id = ''.join(['CV', '0' * (3 - len(cv_number)), cv_number])
            cv = self.toolkit.get_cv(cv_id)

            target_pressure = float(cv.get_field('PVOL'))

            distances.append(abs(target_pressure - obs[i]))
            current_pressures[cv_id] = obs[i]

        return sum(distances), current_pressures

    def __check_termination(self, obs):
        return False
