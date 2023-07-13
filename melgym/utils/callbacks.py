
import os

from pandas import DataFrame
from stable_baselines3.common.callbacks import BaseCallback


class TbMetricsCallback(BaseCallback):
    """
    Custom callback for adding additional logger data.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Last action (normalized)
        norm_action = self.locals['actions'][-1][-1]
        self.logger.record('sim_data/last_norm_action', norm_action)

        # Last action (real)
        real_action = (norm_action + 1) * 10 / 2
        self.logger.record('sim_data/last_real_action', real_action)

        return True


class EpisodicDataCallback(BaseCallback):
    """
    Custom callback for registering episode information.
    """

    def __init__(self, verbose=0, save_freq=50, save_path='ep_metrics/'):
        super().__init__(verbose)
        self.ep_data = []
        self.step_data = {}
        self.n_steps = 0
        self.save_freq = save_freq
        self.save_path = save_path

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def _on_step(self) -> bool:
        # Get step data
        self.step_data['tend'] = self.locals['infos'][-1]['time']        
        for key, value in self.locals['infos'][-1]['pressures'].items():
            self.step_data['pressures-' + key] = value
        for key, value in self.locals['infos'][-1]['distances'].items():
            self.step_data['distances-' + key] = value
        self.step_data['reward'] = self.locals['rewards']
        self.step_data['action'] = self.locals['actions']

        self.ep_data.append(self.step_data)
        self.step_data= {}

        # Periodic saving in .csv
        if not self.n_steps % self.save_freq:
            DataFrame(self.ep_data).to_csv(
                self.save_path + 'episode-' + str(self.n_steps) + '.csv')  
            self.ep_data = []

        self.n_steps += 1

        return True