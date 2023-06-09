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
    id='simple-v0',
    entry_point='melgym.envs:EnvHVAC',
    kwargs={
        'input_file': 'simple.inp',
        'n_actions': 1,
        'controlled_cvs': ['CV001', 'CV006']
    }
)

register(
    id='branch-v0',
    entry_point='melgym.envs:EnvHVAC',
    kwargs={
        'input_file': 'branch.inp',
        'n_actions': 1,
        'controlled_cvs': ['CV086', 'CV083', 'CV026', 'CV011', 'CV006', 'CV001']
    }
)
