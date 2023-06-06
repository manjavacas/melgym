import os
import shutil
import subprocess
import numpy as np

from gymnasium import Env, spaces

from ..utils.constants import *


class EnvHVAC(Env):
    """
    HVAC control environment. Inlet air velocities are adjusted according to HVAC served room states.
    """

    def __init__(self, n_obs, n_actions, control_horizon=100):
        """
        Class constructor.

        Args:
            n_obs (int): number of HVAC served rooms.
            n_actions (int): number of inlet air velocities to be controlled.
            control_horizon (int, optional): number of simulation cycles. Defaults to 100.
        """

        # Observation space
        low_obs = np.zeros(n_obs)
        high_obs = np.inf * np.ones(n_obs)
        self.observation_space = spaces.Box(
            low=low_obs, high=high_obs, dtype=np.float64)

        # Action space
        low_act = np.zeros(n_actions)
        high_act = MAX_DUCT_VELOCITY * np.ones(n_actions)
        self.action_space = spaces.Box(
            low=low_act, high=high_act, dtype=np.float64)

        # Auxiliar variables
        self.control_horizon = control_horizon
        self.edf_path = None
        self.steps_counter = 0

    def reset(self):
        """
        Environment reset method.
            1. Cleans past simulations outputs.
            2. Generates a copy from original input file.
            3. Updates simulation times according to the specified control horizon.
            4. Executes MELGEN, generating an initial restart file.
            5. Executes MELCOR, creating the EDF file.
            6. Reads the first observation from the last EDF record.

        Raises:
            Exception: if an EDF file is not found in the output directory.

        Returns:
            (np.array, dict): first observation and additional information.
        """

        # Env setup
        self.__clean_out_files()
        shutil.copy(INPUT_PATH, MELIN_PATH)
        self.steps_counter += 1
        self.__edit_tend(MELIN_PATH)

        # First simulation
        subprocess.call([MELGEN_PATH, MELIN_PATH], cwd=OUTPUT_DIR)
        subprocess.call([MELCOR_PATH, MELIN_PATH], cwd=OUTPUT_DIR)

        # Get initial results
        edf_files = [os.path.join(OUTPUT_DIR, file) for file in os.listdir(
            OUTPUT_DIR) if file.endswith('.DAT')]
        self.edf_path = edf_files[0] if edf_files else None

        if self.edf_path:
            time, obs = self.__get_edf_obs()
        else:
            raise Exception(''.join(['EDF not found in ', OUTPUT_DIR]))

        return (obs, {'time': time})

    def step(self):
        pass

    def render(self):
        pass

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

    def __edit_tend(self, melin_path):
        """
        Updates the TEND register in the input file according to the specified control horizon.

        Args:
            melin_path (str): path to the input file to be edited.

        Raises:
            Exception: if no TEND register is found in the input file.
            Exception: if a  step is performed without an initial call to reset.
        """
        if self.steps_counter > 0:
            with open(melin_path, 'r+') as f:
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
                        ''.join(['TEND not especified in ', INPUT_PATH]))
            self.steps_counter += 1
        else:
            raise Exception('No reset has been done before step')

    def __get_edf_obs(self):
        """
        Get the last recorded values in the simulation EDF.

        Returns:
            (dict, np.array): last registered values and its corresponding cycle.
        """
        with open(self.edf_path, 'r') as f:
            last_record = f.readlines()[-1].split()
            time = float(last_record[0])
            values = np.array(last_record[1:], dtype=np.float64)
        return (time, values)
