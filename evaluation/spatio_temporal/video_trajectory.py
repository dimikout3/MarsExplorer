import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import json
import os
import pickle as p

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plotstyle="ggplot"
plt.style.use(f"{plotstyle}")

LEVELS = [
"1_1_1", "1_2_1", "1_1_2", "1_2_2",
"2_1_1", "2_2_1", "2_1_2", "2_2_2",
"3_1_1", "3_2_1", "3_1_2", "3_2_2"]
SAVE = True
TITLE_LABEL=True


def generate_cmap(ncolors=256, basic_color=[1.,0.,0.,1.]):
    # import pdb; pdb.set_trace()
    if ncolors<2:ncolors=2

    color_array = np.array([basic_color]*ncolors)
    # change alpha values
    color_array[:,-1] = np.linspace(0.0,0.8,ncolors)
    # create a colormap object
    map_object = LinearSegmentedColormap.from_list(name='custom',colors=color_array)
    # register this new colormap with matplotlib
    plt.register_cmap(cmap=map_object)


def combined(data, title, basic_colors):

    img = plt.imread("mars.jpg")
    plt.imshow(img, extent=[0, 21, 0, 21])

    for id, heat_map in enumerate(data):

        generate_cmap(ncolors = np.max(heat_map).astype(int),
                      basic_color = basic_colors[id])

        plt.imshow(heat_map, cmap='custom')

    plt.yticks([])
    plt.xticks([])

    plt.ylabel('')
    plt.xlabel('')
    plt.grid(False)

    plt.xlim(0,21)
    plt.ylim(0,21)
    plt.gca().invert_yaxis()

    if SAVE:
        plt.savefig(f"images/frame_{frame}.png", bbox_inches='tight', dpi=300)
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":

    # for level in LEVELS:
    # path = f"rollout_{level}.pkl"
    path='rollout_2_2_2.pkl'
    rollouts = p.load(open(path,'rb'))

    x_data = []
    y_data = []
    trajectory = np.zeros((21, 21))
    terrain = np.zeros((21, 21))

    frame = 0

    for rollout in rollouts:
        for time_i, time_step in enumerate(rollout):

            if time_i>50:continue

            x, y, chanel = np.where(time_step[0] == 0.6)
            x_data.append(np.int(x))
            y_data.append(np.int(y))
            trajectory[y,x] += 1

        x, y, chanel = np.where(time_step[0] == 1.0)
        terrain[y,x] += 1

        frame += 1
        combined([trajectory, terrain], f"combined_{frame}",
                 [[.0, .0, 1., 1.], [.0, 1., .0, 1.]])
