import numpy as np

from melgym.envs.melcor import MelcorEnv


class PressureEnv(MelcorEnv):
    """
    Presure control environment.

    This subclass implements the reward, termination, truncation, and rendering methods
    for a pressure control environment.
    """

    def render(self):
        """
        Renders the environment.

        In this example, rendering simply prints the current simulation time and controlled values.
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
        return False

    def _check_truncation(self, obs, info):
        """
        Checks if the episode should be truncated.

        Args:
            obs (np.array): Current observation.
            info (dict): Additional information about the current state.

        Returns:
            bool: True if the episode should be truncated, False otherwise.
        """
        return float(info['TIME']) > 100.0
