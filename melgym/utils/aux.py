def summary(episode, action, obs, reward, info, min_velocity=0, max_velocity=10):
    """
    Prints an episode summary.

    Args:
        episode (int): current episode number.
        action (np.array): last performed action.
        obs (np.array): last observation.
        reward (float): last reward.
        info (dict): additional information.
        max_velocity (float): max action value. Defaults to 10.
        min_velocity (float): min action value. Defaults to 0.
    """
    print(''.join(['\n', 80 * '-', '\nEpisode ', str(episode), 2 * '\n',
                   '[ACTION (norm)] ', str(action), '\n',
                   '[ACTION (real)] ', str(
                       ((action + 1) * ((max_velocity - min_velocity) / 2)) + min_velocity), '\n',
                   '\n[REWARD] ', str(reward), '\n[OBSERVATION] ', str(obs),
                   '\n[TIME] ', str(info['time']), '\n[PRESSURES] ', str(
                       info['pressures']),
                   '\n[DISTANCES] ', str(info['distances']), '\n', 80 * '-']))
