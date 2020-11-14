import numpy as np
import matplotlib.pyplot as plt
import argparse
from randomMapGenerator import Generator
from lidarSensor import Lidar

STEPS = 10

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--width", help="Map width")
    parser.add_argument("-t", "--height", help="Map height")
    parser.add_argument("-n", "--number", help="Number of obstacles")
    args = parser.parse_args()

    # print(f"W={args.width} H={args.height} N={args.number}")
    gen = Generator(size=[30,30],
                    number_rows=3, number_columns=3,
                    noise=[0.04,0.04],
                    margins=[0.2, 0.2],
                    obstacle_size=[0.1, 0.1])

    ldr = Lidar(r=6, channels=32)

    for step in range(STEPS):

        ego_position = np.array([10,10+step])

        map = gen.get_map()

        thetas, ranges = ldr.scan(map, ego_position)

        xObs = (ego_position[0]+ranges*np.cos(thetas)).astype(float)
        yObs = (ego_position[1]+ranges*np.sin(thetas)).astype(float)
        plt.scatter(yObs, xObs, c='r', alpha=0.6)

        for x,y in zip(xObs, yObs):
            plt.plot([y,ego_position[1]], [x, ego_position[0]],
                     c='r', linewidth=1, alpha=0.6)

        plt.imshow(map)
        # plt.show()
        plt.savefig(f"step_{step}.png")
        plt.close()
