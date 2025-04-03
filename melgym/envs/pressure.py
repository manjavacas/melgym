import numpy as np

from melgym.envs.melcor import MelcorEnv


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

    def __init__(self, melcor_model, control_cfs, min_action_value, max_action_value,
                 setpoints=None, max_deviation=1e4, max_episode_len=1e2, warmup_time=100):
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

    def render(self):
        """
        Renders the environment.
        """
        try:
            values = self._get_last_edf_data()
            print("\nRender:")
            print(f"\tTIME: {values[0]}")
            print("\tControlled values:", values[1:])
        except Exception as e:
            print(f"Render error: {e}")

    def _compute_reward(self, obs, info):
        """
        Computes the reward based on the current state.

        Args:
            obs (np.array): Current observation.
            info (dict): Additional information about the current state.

        Returns:
            float: Computed reward.
        """
        return -float(np.sum(obs))

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
