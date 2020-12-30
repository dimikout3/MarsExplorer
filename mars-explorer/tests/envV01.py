import gym
import numpy as np
import math
import time
from mars_explorer.envs.settings import DEFAULT_CONFIG as conf

conf["number_rows"] = 3
conf["number_columns"] = 3
conf["noise"] = [2, 2]
conf["margins"] = [3, 3]
conf["obstacle_size"] = [3,3]

conf["viewer"]["night_color"] = (250,250,250)

env = gym.make('mars_explorer:exploConf-v01', conf=conf)
observation = env.reset()

for step in range(3):

    img = env.reset()
    rend = env.render()
    time.sleep(2)

env.close()
