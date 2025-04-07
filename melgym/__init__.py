"""
MELGYM module.
"""

from gymnasium.envs.registration import register

register(
    id='pressure-v0',
    entry_point='melgym.envs.pressure:PressureEnv',
    kwargs={
        'melcor_model': 'melgym/data/pressure.inp',
        'control_cfs': ['CF007'],
        'min_action_value': 0.0,
        'max_action_value': 5.0,
        'max_deviation': 1e4,
        'max_episode_len': 500,
        'setpoints': [101000.0]
    }
)