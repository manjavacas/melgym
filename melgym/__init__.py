from gymnasium.envs.registration import register

register(
    id='hvac-v0',
    entry_point='melgym.envs:EnvHVAC'
)