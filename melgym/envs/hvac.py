import os
import shutil
import subprocess

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from math import ceil

from gymnasium import Env, spaces
from melkit.toolkit import Toolkit

from ..utils.constants import *


class EnvHVAC(Env):
    """
    HVAC control environment. Inlet air velocities are adjusted according to HVAC served room states.
    """
    metadata = {'render_modes': ['pressures', 'distances']}

    def __init__(self, input_file, n_actions, controlled_cvs, control_horizon=10, check_done_time=500, max_deviation=40, min_velocity=0, max_velocity=10, max_tend=10_000, render_mode=None, ylims=(101000, 101350), time_bt_frames=0.01, env_id=None):
        """
        Class constructor.

        Args:
            input_file (str): name of the file with the MELGEN/MELCOR input data.
            n_actions (int): number of inlet air velocities to be controlled.
            controlled_cvs (list): list of controlled CVs IDs (e.g. CV001).
            control_horizon (int, optional): number of simulation cycles between actions. Defaults to 10.
            check_done_time (int, optional): simulation time allowed before evaluating the termination condition. Defaults to 500.
            max_deviation (float, optional): maximum distance allowed from original pressures. Defaults to 20 (Pa).
            min_velocity (float, optional): minimum value for control actions. Defaults to 0 (m/s).
            max_velocity (float. optional): maximum value for control actions. Defaults to 10 (m/s).
            max_tend (int, optional): maximum TEND before truncation. Defaults to 10000.
            render_mode (str, optional): render option.
            ylims (tuple, optional): ylims for plotting.
            time_bt_frames (float, optional): time between rendered frames. Defaults to 0.01.
            env_id (str, optional): custom environment identifier. Used for naming the output directory.
        """

        # Observation space
        self.controlled_cvs = controlled_cvs
        n_obs = len(self.controlled_cvs)

        self.observation_space = spaces.Box(
            low=np.zeros(n_obs), high=np.inf * np.ones(n_obs), dtype=np.float32)

        # Normalized action space [-1, 1]
        self.action_space = spaces.Box(
            low=-np.ones(n_actions), high=np.ones(n_actions), dtype=np.float32)

        # Environment ID and output directory
        self.env_id = 'melgym_' + \
            datetime.now().strftime('%Y%m%d_%H:%M:%S') if not env_id else env_id
        self.output_dir = os.path.join(OUTPUT_DIR, self.env_id)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Paths
        self.input_path = os.path.join(DATA_DIR, input_file)
        self.melin_path = os.path.join(self.output_dir, 'MELIN')
        self.melog_path = os.path.join(self.output_dir, 'MELOG')
        self.edf_path = None

        # Control variables
        self.control_horizon = control_horizon
        self.check_done_time = check_done_time
        self.max_deviation = max_deviation
        self.max_velocity = max_velocity
        self.min_velocity = min_velocity
        self.max_tend = max_tend

        # Aux variables
        self.n_steps = 0
        self.current_tend = 0

        # Tookit
        self.toolkit = Toolkit(self.input_path)

        # Initial pressures
        self.init_pressures = {cv_id: self.__get_initial_pressure(
            cv_id) for cv_id in self.controlled_cvs}

        # Render
        if render_mode == 'pressures':

            self.render_mode = render_mode
            self.time_bt_frames = time_bt_frames

            plt.ion()

            _, self.axs = plt.subplots(1, 2, figsize=(14, 8))

            # set initial x,y values
            self.x = [self.current_tend]
            self.ys = []
            for cv, pressure in self.init_pressures.items():
                self.ys.append({'cv': cv, 'values': [pressure]})

            # create plots
            self.graphs = []
            for record in self.ys:
                self.graphs.append(
                    self.axs[0].plot(self.x, record['values'], label=record['cv'])[0])

            # add hlines with ISO classes
            iso_classes = [(101065, 'C4'), (101215, 'C3'),
                           (101235, 'C2'), (101295, 'C1')]
            for pressure, iso_class in iso_classes:
                self.axs[0].axhline(y=pressure, color='black',
                                    linestyle='--', linewidth=0.5)
                self.axs[0].text(9.5, pressure, iso_class, color='black',
                                 fontsize=10, ha='left', va='bottom')

            # ajust legend and ylims
            self.axs[0].legend(loc='upper center', bbox_to_anchor=(
                0.5, 1.15), ncol=4, fancybox=True)
            self.axs[0].set_ylim(ylims)
 
            plt.pause(self.time_bt_frames)
        
        else:
            raise Exception('The specified render format is not available.')

    def reset(self, seed=None, options=None):
        """
        Environment initialization.
            1. Generates a copy from original input file.
            2. Updates the simulation time according to the specified control horizon.
            3. Executes MELGEN, generating an initial restart file.
            4. Gets the initial pressures for each CV.
            5. CFs redefinitions are added to the MELCOR input for future control steps.

        Args:
            seed (int, optional): random seed. Defaults to None.
            options (dict, optional): reset options. Defaults to None.

        Returns:
            np.array: initial pressures.
            dict: initial time step and pressures.
        """

        # Env setup
        self.__clean_out_files()
        shutil.copy(self.input_path, self.melin_path)

        self.n_steps = 1

        self.__update_time()

        # MELGEN execution
        with open(self.melog_path, 'a') as log:
            subprocess.call([MELGEN_PATH, self.melin_path],
                            cwd=self.output_dir, stdout=log, stderr=subprocess.STDOUT)

        # Get initial pressures
        info = {'time': 0.0}
        for cv_id in self.controlled_cvs:
            info[cv_id] = self.init_pressures[cv_id]
        obs = [value for key, value in info.items()
               if key in self.controlled_cvs]

        # Add CFs redefinition to MELCOR input
        self.__add_cfs_redefinition()

        return np.array(obs, dtype=np.float32), info

    def step(self, action):
        """
        Performs a control action by adjusting inlet air velocities.
            1. Modifies input CFs related to inlet FL velocities.
            2. Updates TEND.
            3. Executes a MELCOR simulation during the number of cycles specified in the control horizon.
            4. Gets the last values recorded in the EDF.
            5. Computes the reward based on pressure requirements.
            6. Evaluates the episode termination and truncation condition.

        Args:
            action (np.array): new inlet air velocities.

        Returns:
            np.array: environment observation after control action (current controlled pressures).
            float: reward value. Computed as the sum of the distances of every room pressure to its initial value.
            bool: termination flag. It will be True if any pressure exceeds its imposed limits.
            bool: truncated flag. It will be True if pressures keep stable until the imposed time step limit.
            dict: additional step information (last recorded time and pressures).

        Raises:
            Exception: if step() is called without an initial call to reset().
        """

        # Denormalize action
        action = self.__denorm_action(action)

        # Input edition
        if self.n_steps > 0:
            self.__update_time()
        else:
            raise Exception('Error: reset() has not been called before step()')

        self.__update_cfs(action)

        # MELCOR simulation
        with open(self.melog_path, 'a') as log:
            subprocess.call([MELCOR_PATH, 'ow=o', 'i=' + self.melin_path],
                            cwd=self.output_dir, stdout=log, stderr=subprocess.STDOUT)
        # Get results
        time, pressures = self.__get_last_record()
        obs = [value for key, value in pressures.items()
               if key in self.controlled_cvs]
        reward, distances = self.__compute_distances(pressures)

        info = {**time, 'pressures': pressures, 'distances': distances}

        # Check ending conditions
        terminated = self.__check_termination(distances)
        truncated = self.__check_truncation()

        self.n_steps += 1

        return np.array(obs, np.float32), -reward, terminated, truncated, info

    def render(self):
        """
        Plots all pressures / distances evolution during simulation time.
        """
        if self.render_mode == 'pressures' and self.n_steps > 1:

            # Plot 1 (pressure evolution)
            self.x.append(self.current_tend)
            current_pressures = self.__get_last_record()[1]

            for y, graph in zip(self.ys, self.graphs):
                y['values'].append(current_pressures[y['cv']])
                graph.set_data(self.x, y['values'])

            self.axs[0].set_xlim(self.x[0], self.x[-1])
            # also: plt.xlim(self.x[-2], self.x[-1])

            # Plot 2 (pressure distance)            
            _, distances = self.__compute_distances(current_pressures)
            cv_ids = [cv_id for cv_id in distances.keys()]
            distances= [distance for distance in distances.values()]
            
            self.axs[1].clear()
            self.axs[1].bar(cv_ids, distances, color='r')
            
            self.axs[1].set_xlabel('CV')
            self.axs[1].set_ylabel('Distance')
            self.axs[1].set_xticklabels(cv_ids, rotation=45)

            plt.pause(self.time_bt_frames)

    def close(self):
        """
        Cleans every output file except PTFs (*PTF) and EDFs (*.DAT).
        """
        for file in os.listdir(self.output_dir):
            if not file.endswith(('PTF', '.DAT')):
                os.remove(os.path.join(self.output_dir, file))

    def __clean_out_files(self):
        """
        Cleans the output directory where past simulation files are stored.
        """
        for file in os.listdir(self.output_dir):
            os.remove(os.path.join(self.output_dir, file))

    def __update_time(self):
        """
        Edits the TEND register in the input file according to the specified control horizon.

        Args:
            melin_path (str): path to the input file to be edited.

        Raises:
            Exception: if no TEND register is found in the input file.
        """

        with open(self.melin_path, 'r+') as f:
            lines = f.readlines()
            edit_line = -1
            for i, line in enumerate(lines):
                # Get TEND line
                if 'TEND' in line:
                    edit_line = i
                    break
            if edit_line != -1:
                # Set new TEND
                new_tend = str(
                    self.control_horizon * self.n_steps) if self.n_steps > 1 else str(self.control_horizon)
                lines[edit_line] = ''.join(['TEND ', new_tend, '\n'])
                self.current_tend = int(new_tend)
                # Update file with new TEND
                f.seek(0)
                f.writelines([line.strip() + '\n' for line in lines])
                f.truncate()
            else:
                raise Exception(
                    ''.join(['TEND not especified in ', self.input_path]))

    def __denorm_action(self, action):
        """
        Maps [-1,1] actions to [min_velocity, max_velocity]

        Args:
            action (np.array): current normalized action.

        Returns:
            np.array: denormalized action.
        """
        return ((action + 1) * ((self.max_velocity - self.min_velocity) / 2)) + self.min_velocity

    def __get_last_record(self):
        """
        Reads the last recorded values in an EDF.

        Raises:
            Exception: if the EDF is not propertly read.

        Returns:
            (dict, dict): last registered time and values (i.e. pressures).
        """

        time = {}
        record = {}

        n_cvs = len(self.controlled_cvs)

        with open(self.__get_edf_path(), 'r') as f:
            try:
                lines = f.readlines()
                last_registered = []
                for line in reversed(lines):
                    last_registered = line.split() + last_registered
                    if len(last_registered) >= n_cvs + 1:
                        break
            except:
                raise Exception('Error reading empty EDF.')

            time['time'] = float(last_registered[0])
            for i, cv_id in enumerate(self.controlled_cvs):
                record[cv_id] = np.float32(last_registered[i+1])

        return time, record

    def __add_cfs_redefinition(self, cf_keyword='CONTROLLER'):
        """
        Includes the definitions of the CFs to be overwritten in the MELCOR input.

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
        Computes the sum of the distances of each controlled CV pressure to their initial (target) value.

        Args:
            last_record (dict): last recorded data in the EDF.

        Returns:
            float: sum of all the distances between CV pressures to their initial values.
            dict: individual pressure distances.
        """

        distances = {}

        ids = [id for id in last_record if id in self.controlled_cvs]

        for cv_id in ids:
            distances[cv_id] = last_record[cv_id] - \
                self.init_pressures[cv_id]

        # return sum(pow(abs(value), 2) for value in distances.values()), distances
        return sum(abs(value) for value in distances.values()), distances

    def __get_initial_pressure(self, cv_id):
        """
        Returns the initial pressure of a CV.

        Args:
            cv_id (str): CV identifier (CVnnn).

        Returns:
            float: initial pressure.
        """

        return float(self.toolkit.get_cv(cv_id).get_field('PVOL'))

    def __get_edf_path(self):
        """
        Returns the path to the EDF generated by MELCOR.
        In case multiple EDFs exist in the same output directory, the first EDF found will be used.

        Raises:
            Exception: if the EDF is not found.

        Returns:
            str: path to the EDF.
        """

        edf_files = [os.path.join(self.output_dir, file) for file in os.listdir(
            self.output_dir) if file.endswith('.DAT')]
        self.edf_path = edf_files[0] if edf_files else None

        if not self.edf_path:
            raise Exception(''.join(['EDF not found in ', self.output_dir]))

        return self.edf_path

    def __check_termination(self, distances):
        """
        Checks the completion of an episode. 
        An episode ends when at least one pressure has moved k Pa away from its original value, where k is the class variable max_deviation.
        The termination condition will be evaluated when current_tend >= check_done_time.

        Args:
            distances (dict): computed distances from original to current pressures.

        Returns:
            bool: True if at least one pressure is out of its allowed limits. False if termination is not yet evaluable or if pressures are inside their limits.
        """

        return any(abs(value) > self.max_deviation for value in distances.values()) and self.current_tend > self.check_done_time

    def __check_truncation(self):
        """
        Checks the truncation condition of an episode. Waits until check_done_time is raised to be evaluated.
        An episode is truncated if the number of steps is greater than a value specified in max_tend.

        Returns:
            bool: True if the maximum number of steps have been raised. False if not, or if truncation is not yet evaluable.
        """

        return self.current_tend > self.max_tend and self.current_tend > self.check_done_time
