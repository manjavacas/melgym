"""
MELGYM module.
"""

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
    id='presscontrol',
    entry_point='melgym.envs:EnvPress',
    kwargs={
        'input_file': 'presscontrol.inp',
        'n_actions': 1,
        'controlled_cvs': ['CV002']
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

register(
    id='branch_3',
    entry_point='melgym.envs:EnvHVAC',
    kwargs={
        'input_file': 'branch_3.inp',
        'n_actions': 1,
        'controlled_cvs': ['CV002', 'CV010', 'CV014', 'CV017', 'CV018', 'CV024', 'CV053', 'CV056', 'CV057', 'CV058', 'CV095']
    }
)
