import csv
import os
import numpy as np
import matplotlib.pyplot as plt

from melgym.envs.melcor import MelcorEnv


class PressureEnv(MelcorEnv):
    """
    Pressure control environment.

    This subclass re-implements the reset, saatep, compute_reward, check_termination, check_truncation, and rendering methods.
    for a pressure control environment.

    Args:
        melcor_model (str): Path to the MELCOR model file.
        control_cfs (list): List of controlled CFs.
        min_action_value (float): Minimum action value.
        max_action_value (float): Maximum action value.
        setpoints (list): List of setpoints.
        max_deviation (float): Maximum deviation from setpoints before truncation.
        max_episode_len (float): Maximum length of an episode before truncation.
        render_mode (str): Render mode. Default is None.
        logging (bool): Logging option. Default is False.
    """
    metadata = {
        "render_modes": ['human'],
        "render_fps": 30
    }

    def __init__(self, melcor_model, control_cfs, min_action_value, max_action_value,
                 setpoints, max_deviation, max_episode_len, render_mode=None, logging=False):
        super().__init__(melcor_model=melcor_model, control_cfs=control_cfs,
                         min_action_value=min_action_value, max_action_value=max_action_value)

        self.setpoints = setpoints
        self.max_deviation = max_deviation
        self.max_episode_len = max_episode_len

        if render_mode:
            self.render_mode = render_mode
        self.time_data = []
        self.obs_data = []

        # CSV logging setup
        self.logging = logging
        if self.logging:
            self.log_path = "log.csv"
            self._init_csv_log()

    def reset(self, **kwargs):
        """
        Resets the environment and clears the plot.

        Returns:
            tuple: A tuple containing the initial observation (i.e., provided setpoints) and env information.
        """
        obs, info = super().reset(**kwargs)

        plt.clf()
        self.time_data.clear()
        self.obs_data.clear()

        obs = np.array(self.setpoints)

        if self.logging:
            self._write_csv_log(
                step=self.n_steps,
                action="",
                obs=obs,
                reward="",
                termination="",
                truncation=""
            )

        return obs, info

    def step(self, action):
        """
        Redefinition of the step method to allow logging.

        Returns:
            tuple: A tuple containing the observation, reward, termination, truncation, and env information.
        """
        obs, reward, termination, truncation, info = super().step(action)

        if self.logging:
            self._write_csv_log(
                step=self.n_steps,
                action=action,
                obs=obs,
                reward=reward,
                termination=termination,
                truncation=truncation
            )

        return obs, reward, termination, truncation, info

    def render(self):
        """
        Renders the controlled pressures.
        """
        if self.n_steps > 1:
            try:
                values = self._get_last_edf_data()
                time = values[0]
                controlled_values = values[1:]

                self.time_data.append(time)
                self.obs_data.append(controlled_values)

                self._update_plot()
            except Exception as e:
                print(f"Render error: {e}")

    def _update_plot(self):
        """
        Updates the pressures plot.
        """
        plt.clf()
        for i, value in enumerate(np.array(self.obs_data).T):
            plt.plot(self.time_data, value, label=self.controlled_values[i])
        plt.xlabel("Timestep")
        plt.ylabel("Pressure (Pa)")
        plt.legend()
        plt.pause(0.1)

    def _compute_reward(self, obs, info):
        """
        Computes the reward based on the distance between the current observation and the given setpoints.

        Args:
            obs (np.array): Current observation.
            info (dict): Additional information about the current state.

        Returns:
            float: Computed reward.
        """
        return -np.mean(np.abs(np.array(self.setpoints) - obs))

    def _check_termination(self, obs, info):
        """
        Checks if the episode has terminated.

        Args:
            obs (np.array): Current observation.
            info (dict): Additional information about the current state.

        Returns:
            bool: True if the episode should terminate, False otherwise.
        """
        return False

    def _check_truncation(self, obs, info):
        """"
        Checks if the episode should be truncated based on time limit or maximum allowed deviation.

        Args:
            obs (np.array): Current observation.
            info (dict): Additional information about the current state.

        Returns:
            bool: True if the episode should be truncated, False otherwise.
        """
        time_limit = bool(info['TIME'] >= self.max_episode_len)
        press_limit = bool(np.any(np.abs(obs - np.array(self.setpoints)) > self.max_deviation))

        return time_limit or press_limit

    def _init_csv_log(self):
        """
        Initializes the CSV log file with headers.
        """
        if not os.path.exists(self.log_path):
            with open(self.log_path, mode='w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "step", "action", "obs", "reward", "termination", "truncation"
                ])
                writer.writeheader()

    def _write_csv_log(self, step, action, obs, reward, termination, truncation):
        """
        Appends a row to the CSV log file.
        """
        with open(self.log_path, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                "step", "action", "obs", "reward", "termination", "truncation"
            ])
            writer.writerow({
                "step": step,
                "action": action.tolist() if isinstance(action, np.ndarray) else action,
                "obs": obs.tolist() if isinstance(obs, np.ndarray) else obs,
                "reward": reward,
                "termination": termination,
                "truncation": truncation
            })
            if termination or truncation:
                writer.writerow({
                    "step": "",
                    "action": "",
                    "obs": "",
                    "reward": "",
                    "termination": "",
                    "truncation": ""
                })
