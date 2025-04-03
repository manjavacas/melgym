"""
MELGYM module.
"""

from gymnasium.envs.registration import register

register(
    id='pressure',
    entry_point='melgym.envs.pressure:PressureEnv',
    kwargs={
        'melcor_model': 'melgym/data/pressure.inp',
        'control_cfs': ['CF007'],
        'min_action_value': 0.0,
        'max_action_value': 10.0
    }
)