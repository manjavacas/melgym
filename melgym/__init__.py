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
    id='branch-nl-v0',
    entry_point='melgym.envs:EnvHVAC',
    kwargs={
        'input_file': 'branch-nl.inp',
        'n_actions': 1,
        'controlled_cvs': ['CV086', 'CV083', 'CV026', 'CV011', 'CV006', 'CV001']
    }
)
