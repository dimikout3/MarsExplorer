import numpy as np
import matplotlib.pyplot as plt
import argparse
from randomMapGenerator import Generator
from lidarSensor import Lidar


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--width", help="Map width")
    parser.add_argument("-t", "--height", help="Map height")
    parser.add_argument("-n", "--number", help="Number of obstacles")
    args = parser.parse_args()

    ego_position = np.array([0,0])

    # print(f"W={args.width} H={args.height} N={args.number}")
    gen = Generator(size=[30,30],
                    number_rows=3, number_columns=3,
                    noise=[0.04,0.04],
                    margins=[0.2, 0.2],
                    obstacle_size=[0.1, 0.2])

    ldr = Lidar(r=25, channels=1024)
    map = gen.get_map()

    thetas, ranges = ldr.scan(map, ego_position)
    print(ranges)
    # import pdb; pdb.set_trace()

    xObs = (ego_position[0]+ranges*np.cos(thetas)).astype(int)
    yObs = (ego_position[1]+ranges*np.sin(thetas)).astype(int)
    plt.scatter(yObs, xObs)
    # plt.show()
    # plt.close()

    plt.imshow(map)
    plt.show()
    plt.close()
