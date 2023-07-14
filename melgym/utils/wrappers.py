from gymnasium import ObservationWrapper, ActionWrapper, RewardWrapper
from gymnasium import spaces

import numpy as np


class NormalizedObservations(ObservationWrapper):
    """
    Observation normalization. Maps observations to range [0,1].
    """

    def __init__(self, env, min_val, max_val):
        super(ObservationWrapper, self).__init__(env)
        self.min_val = min_val
        self.max_val = max_val
        self.observation_space = spaces.Box(
            low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        # Normalize in [0, 1]
        return (observation - self.min_val) / (self.max_val - self.min_val)


class DenormalizedActions(ActionWrapper):
    """
    Action denormalization. Maps [-1,1] to [min_val, max_val].
    """

    def __init__(self, env, min_val, max_val):
        super(ActionWrapper, self).__init__(env)
        self.min_val = min_val
        self.max_val = max_val
        self.action_space = spaces.Box(
            low=min_val, high=max_val, shape=env.action_space.shape, dtype=np.float32)

    def action(self, action):
        # Map [-1, 1] to [min_val, max_val]
        return (action + 1) * ((self.max_val - self.min_val) / 2) + self.min_val