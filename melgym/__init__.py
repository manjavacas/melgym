from gymnasium.envs.registration import register

register(
    id='base-v0',
    entry_point='melgym.envs:EnvHVAC',
    kwargs={
        'input_file': 'base.inp',
        'n_actions': 1,
        'controlled_cvs': ['CV003']
    }
)

register(
    id='branch0-v0',
    entry_point='melgym.envs:EnvHVAC',
    kwargs={
        'input_file': 'branch_0.inp',
        'n_actions': 1,
        'controlled_cvs': ['CV001', 'CV006', 'CV011', 'CV026', 'CV083', 'CV086']
    }
)
