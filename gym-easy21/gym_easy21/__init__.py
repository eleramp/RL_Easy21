from gym.envs.registration import register

register(
        id='easy21-v0',
        entry_point='gym_easy21.envs:Easy21Env',
)
