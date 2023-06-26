from stable_baselines3.common.callbacks import BaseCallback


class MetricsCallback(BaseCallback):
    """
    Custom callback for gathering additional simulation data.
    """

    def __init__(self, verbose=0, csv_save=False, metrics_folder='metrics', log_freq=1):
        super().__init__(verbose)
        self.csv_save = csv_save
        if self.csv_save:
            import os
            self.metrics_folder = metrics_folder
            self.log_freq = log_freq
            if not os.path.exists(self.metrics_folder):
                os.makedirs(self.metrics_folder)

    def _on_step(self) -> bool:

        ################# Tensorboard #################

        # Last action (normalized)
        norm_action = self.locals['actions'][-1][-1]
        self.logger.record('sim_data/last_norm_action', norm_action)

        # Last action (real)
        real_action = (norm_action + 1) * 10 / 2
        self.logger.record('sim_data/last_real_action', real_action)

        ############## .CSV episode data ##############

        if self.csv_save:
            episode = self.training_env.get_attr('n_episodes')[-1]
            if episode == 1 or episode % self.log_freq == 0:
                with open(self.metrics_folder + '/episode_' + str(episode) + '.csv', 'a') as f:
                    f.write(str(norm_action) + '\t' + str(real_action) + '\n')

        return True
