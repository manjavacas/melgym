from gymnasium.envs.registration import register

register(
    id='sample-v0',
    entry_point='melgym.envs:SampleEnv'
)
