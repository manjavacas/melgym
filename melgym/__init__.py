from gymnasium.envs.registration import register

register(
    id='hvac-v0',
    entry_point='melgym.envs:HVACEnv'
)

register(
    id='sgs-v0',
    entry_point='melgym.envs:SGSEnv'
)