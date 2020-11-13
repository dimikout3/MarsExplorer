import numpy as np
import matplotlib.pyplot as plt
import argparse
from randomMapGenerator import Generator


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--width", help="Map width")
    parser.add_argument("-t", "--height", help="Map height")
    parser.add_argument("-n", "--number", help="Number of obstacles")
    args = parser.parse_args()

    # print(f"W={args.width} H={args.height} N={args.number}")
    gen = Generator(number_rows=3, number_columns=3,
                    noise=[0.04,0.04],
                    margins=[0.25, 0.25],
                    obstacle_size=[0.1, 0.1])

    plt.imshow(gen.get_map())
    plt.show()
    plt.close()
