import numpy as np

from gymnasium import ActionWrapper


class DenormaliseActionsWrapper(ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = env.action_space
        self.real_action_space = env.real_action_space

    def action(self, action):
        """
        Converts normalised actions to their corresponding real values. 
        If the actions are not normalised, they are returned without applying any change.

        Args:
            action (np.array): action array.

        Returns:
            np.array: transformed action values.
        """

        tr_action = np.zeros_like(action)

        for i, value in enumerate(action):
            norm_range = self.action_space.high[i] - self.action_space.low[i]
            real_range = self.real_action_space.high[i] - \
                self.real_action_space.low[i]
            tr_action[i] = self.real_action_space.low[i] + \
                (value - self.action_space.low[i]) * real_range / norm_range

        return tr_action
