import numpy as np
import matplotlib.pyplot as plt
# Include the parent direcotry of GEP in python path (not nice looking)
import os, sys
sys.path.append(os.path.join(os.getcwd(), ".."))

import time

from Envs.lidar_V02 import Grid

TIME_STEPS = 40

if __name__ == "__main__":

    env = Grid(size=[42,42])

    state = env.reset(start=[20,20])
    env.render()

    for step in range(TIME_STEPS):

        action = env.action_space_sample()
        new_state, reward, is_done, _ = env.step(action)

        env.render()
        time.sleep(.3)
        print(f"Time step:{step}")
