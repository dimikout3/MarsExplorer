from gym.envs.registration import register

register(
    id='explorer-v01',
    entry_point='gym_explorer.envs:Explorer',
)
