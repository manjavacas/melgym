"""
Auxiliary tools.
"""

def summary(episode, action, obs, reward, info, min_velocity=0, max_velocity=10) -> None:
    """
    Prints summary information for one environment step.

    Args:
        step (int): current step number.
        action (np.array): last performed action.
        obs (np.array): last observation.
        reward (float): last reward.
        info (dict): additional information.
        min_velocity (float): min action value. Defaults to 0.
        max_velocity (float): max action value. Defaults to 10.
    """


    print(''.join(['\n', 80 * '-', '\nStep ', str(episode), 2 * '\n',
                   '[ACTION (norm)] ', str(action), '\n',
                   '[ACTION (real)] ', str(
                       ((action + 1) * ((max_velocity - min_velocity) / 2)) + min_velocity), '\n',
                   '\n[REWARD] ', str(reward), '\n[OBSERVATION] ', str(obs),
                   '\n[TIME] ', str(info['time']), '\n[PRESSURES] ', str(
                       info['pressures']),
                   '\n[DISTANCES] ', str(info['distances']), '\n', 80 * '-']))
