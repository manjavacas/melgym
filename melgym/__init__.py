from gymnasium.envs.registration import register

register(
    id='hvac-v0',
    entry_point='melgym.envs:EnvHVAC',
    kwargs={
        'input_file' : 'hvac.inp'
    }
)