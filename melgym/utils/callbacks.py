import csv


from stable_baselines3.common.callbacks import BaseCallback


class TensorboardMetricsCallback(BaseCallback):
    """
    Custom callback for gathering additional simulation data.
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
