from gymnasium.envs.registration import register

register(
    id='simple-v0',
    entry_point='melgym.envs:EnvHVAC',
    kwargs={
        'input_file' : 'simple.inp'
    }
)