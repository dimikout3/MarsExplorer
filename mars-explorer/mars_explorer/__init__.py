from gym.envs.registration import register

register(
    id='explorer-v01',
    entry_point='mars_explorer.envs:Explorer',
)

register(
    id='exploConf-v01',
    entry_point='mars_explorer.envs:ExplorerConf',
)
