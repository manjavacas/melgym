import numpy as np

from melgym.envs.melcor import MelcorEnv

import matplotlib.pyplot as plt


class PressureEnv(MelcorEnv):
    """
    Pressure control environment.

    This subclass implements the reward, termination, truncation, and rendering methods
    for a pressure control environment.

    Args:
        melcor_model (str): Path to the MELCOR model file.
        control_cfs (list): List of controlled CFs.
        min_action_value (float): Minimum action value.
        max_action_value (float): Maximum action value.
        setpoints (list): List of setpoints.
        max_deviation (float): Maximum deviation from setpoints.
        max_episode_len (float): Maximum length of an episode before truncation.
        warmup_time (float): Time before checking for termination.
    """

    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 30
    }

    def __init__(self, melcor_model, control_cfs, min_action_value, max_action_value,
                 setpoints, max_deviation=1e3, max_episode_len=1e4, warmup_time=50, render_mode=None):
        """
        Initializes the PressureEnv environment.
        """
        # Initialize the parent class (MelcorEnv) with the appropriate arguments
        super().__init__(melcor_model=melcor_model, control_cfs=control_cfs,
                         min_action_value=min_action_value, max_action_value=max_action_value)

        # Initialize the PressureEnv-specific arguments
        self.setpoints = setpoints
        self.max_deviation = max_deviation
        self.max_episode_len = max_episode_len
        self.warmup_time = warmup_time

        # Rendering
        if render_mode:
            self.render_mode = render_mode
        self.time_data = []
        self.obs_data = []

    def reset(self, **kwargs):
        """
        Resets the environment. Also clears the plot for a fresh start.
        """
        plt.clf()
        self.time_data.clear()
        self.obs_data.clear()

        return super().reset(**kwargs)

    def render(self):
        """
        Renders the environment interactively, displaying the evolution of variables.
        """
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
        Updates the plot with the latest pressure data.
        """
        plt.clf()

        for i, value in enumerate(np.array(self.obs_data).T):
            plt.plot(self.time_data, value, label=self.controlled_values[i])

        plt.title("Pressure evolution")
        plt.xlabel("Timestep")
        plt.ylabel("Pressure (Pa)")
        plt.legend()

        # plt.hlines(y=self.limits, xmin=0, xmax=self.max_episode_len,colors='red', linestyles='--', lw=2)
        # plt.ylim(-1e5, 1e5)

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
        return -np.mean(np.abs(obs - np.array(self.setpoints)))

    def _check_termination(self, obs, info):
        """
        Checks if the episode has terminated.

        Args:
            obs (np.array): Current observation.
            info (dict): Additional information about the current state.

        Returns:
            bool: True if the episode should terminate, False otherwise.
        """
        return np.any(np.abs(obs - np.array(self.setpoints)) > self.max_deviation) and info['TIME'] > self.warmup_time

    def _check_truncation(self, obs, info):
        """
        Checks if the episode should be truncated.

        Args:
            obs (np.array): Current observation.
            info (dict): Additional information about the current state.

        Returns:
            bool: True if the episode should be truncated, False otherwise.
        """
        return info['TIME'] >= self.max_episode_len
