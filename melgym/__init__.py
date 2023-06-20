from gymnasium.envs.registration import register

register(
    id='base-v0',
    entry_point='melgym.envs:EnvHVAC',
    kwargs={
        'input_file' : 'base.inp',
        'n_actions' : 1,
        'controlled_cvs' : ['CV003']
    }
)