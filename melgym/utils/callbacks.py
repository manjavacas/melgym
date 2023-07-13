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
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.ep_data = {}

    def _on_step(self) -> bool:
        self.ep_data['tend'] = self.locals['infos'][-1]['time']
        self.ep_data['pressures'] = self.locals['infos'][-1]['pressures']
        self.ep_data['distances'] = self.locals['infos'][-1]['distances']
        self.ep_data['observation'] = self.locals['new_obs']
        self.ep_data['reward'] = self.locals['rewards']
        self.ep_data['action'] = self.locals['actions']

        print(self.ep_data)
        print('\n')


        return True

    def _on_rollout_end(self) -> None:
        self.ep_data = {}
        print('--------------------------------------------------')