import matplotlib.pyplot as plt
# Include the parent direcotry of GEP in python path (not nice looking)
import os, sys
sys.path.append(os.path.join(os.getcwd(), ".."))

from Envs.lidar_V01 import Grid

TIME_STEPS = 30

if __name__ == "__main__":

    env = Grid()

    state = env.reset()

    plt.imshow(state[0])
    plt.show()
    plt.close()

    for step in range(TIME_STEPS):

        action = env.action_space_sample()
        new_state, reward, is_done = env.step(0)

        plt.imshow(new_state[0])
        plt.show()
        plt.close()
