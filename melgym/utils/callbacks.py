
import os

from pandas import DataFrame
from stable_baselines3.common.callbacks import BaseCallback


class TbMetricsCallback(BaseCallback):
    """
    Custom callback for adding additional logger data.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.n_ep_steps = 0
        self.mean_ep_reward = 0

    def _on_step(self) -> bool:
        # Actions (normalized)
        norm_action = self.locals['actions'][-1][-1]
        self.logger.record('actions/norm_action', norm_action)

        # Actions (real)
        real_action = (norm_action + 1) * 10 / 2
        self.logger.record('actions/real_action', real_action)

        # Pressures
        for key, value in self.locals['infos'][-1]['pressures'].items():
            self.logger.record('pressures/' + key, value)

        # Distances
        for key, value in self.locals['infos'][-1]['distances'].items():
            self.logger.record('distances/' + key, value)

        # Mean episode reward
        self.n_ep_steps += 1
        self.mean_ep_reward = self.locals['rewards'][-1] + (self.mean_ep_reward / self.n_ep_steps)

        if self.locals['dones'][-1]:
            self.logger.record('episodic/mean_ep_reward', self.mean_ep_reward)
            self.n_ep_steps = 0
            self.mean_ep_reward = 0

        return True


class EpisodicDataCallback(BaseCallback):
    """
    Custom callback for registering episode information.
    """

    def __init__(self, save_path, verbose=0, save_freq=10):
        super().__init__(verbose)

        self.save_freq = save_freq
        self.save_path = save_path
        self.num_episodes = 1
        self.ep_data = []
        self.step_data = {}

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def _on_step(self) -> bool:

        # Get step data
        self.step_data['tend'] = self.locals['infos'][-1]['time']
        for key, value in self.locals['infos'][-1]['pressures'].items():
            self.step_data['pressures-' + key] = value
        for key, value in self.locals['infos'][-1]['distances'].items():
            self.step_data['distances-' + key] = value
        self.step_data['reward'] = self.locals['rewards'][-1]
        self.step_data['action'] = self.locals['actions'][-1][-1]
        self.step_data['done'] = self.locals['dones'][-1]

        self.ep_data.append(self.step_data)
        self.step_data = {}

        # Episode end
        if self.locals['dones'][-1] or self.locals['infos'][-1]['TimeLimit.truncated']:
            if not self.num_episodes % self.save_freq or self.num_episodes == 1:
                DataFrame(self.ep_data).to_csv(self.save_path +
                                               'episode-' + str(self.num_episodes) + '.csv')
            self.num_episodes += 1
            self.ep_data = []

        return True
