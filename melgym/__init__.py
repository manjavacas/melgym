from gymnasium.envs.registration import register

register(
    id='foo',
    entry_point='melgym.envs:EnvHVAC',
    kwargs={
        'input_file': 'foo.inp',
        'n_actions': 1,
        'controlled_cvs': ['CV003']
    }
)

register(
    id='branch_1',
    entry_point='melgym.envs:EnvHVAC',
    kwargs={
        'input_file': 'branch_1.inp',
        'n_actions': 1,
        'controlled_cvs': ['CV001', 'CV006', 'CV011', 'CV026', 'CV083', 'CV086']
    }
)

register(
    id='branch_2',
    entry_point='melgym.envs:EnvHVAC',
    kwargs={
        'input_file': 'branch_2.inp',
        'n_actions': 1,
        'controlled_cvs': ['CV015', 'CV023', 'CV024', 'CV054', 'CV080', 'CV081', 'CV082', 'CV089', 'CV096']
    }
)