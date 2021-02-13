import gym
import numpy as np
import math
import time

import cv2

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plotstyle="ggplot"
plt.style.use(f"{plotstyle}")

from mars_explorer.envs.settings import DEFAULT_CONFIG as conf

ONLY_VIDEO = False
SAVE = True
FRAMES = 30
TIME_VIDEO = 15

conf["initial"] = [-2,-2]
conf["number_columns"] = 3
conf["number_rows"] = 3
conf["viewer"]["night_color"] = (255, 255, 255)
conf["lidar_range"] = 0
# level 1-1-1
conf["noise"] = [3,3]
conf["obstacle_size"] = [5,5]


def generate_cmap(ncolors=256, basic_color=[1.,0.,0.,1.]):
    # import pdb; pdb.set_trace()
    if ncolors<2:ncolors=2

    color_array = np.array([basic_color]*ncolors)
    # change alpha values
    color_array[:,-1] = np.linspace(0.0,.9,ncolors)
    # create a colormap object
    map_object = LinearSegmentedColormap.from_list(name='custom',colors=color_array)
    # register this new colormap with matplotlib
    plt.register_cmap(cmap=map_object)


def saveRend(rend):
    plt.rcParams["axes.grid"] = False
    plt.axis('off')
    plt.imshow(rend)
    plt.savefig(f"env/env_{step}.png", bbox_inches='tight')
    plt.close()


def plotHeatMap(terrain):

    mars = plt.imread("img/mars.jpg")
    plt.imshow(mars, extent=[0, 21, 0, 21], origin='lower')

    generate_cmap(ncolors = np.max(terrain).astype(int))

    plt.imshow(terrain,
               cmap='custom')

    plt.yticks([])
    plt.xticks([])

    plt.ylabel('')
    plt.xlabel('')
    plt.grid(False)

    plt.xlim(0,21)
    plt.ylim(0,21)
    plt.gca().invert_yaxis()

    if SAVE:
        plt.savefig(f"map/heatmap_{step}.png", bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def video_env(time=TIME_VIDEO):
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    image_frame = cv2.imread('env/env_0.png')
    width, height, _ = image_frame.shape
    out = cv2.VideoWriter('env/env.avi', fourcc, FRAMES/time, (width,height))

    for i in range(FRAMES):
        image_frame = cv2.imread(f"env/env_{i}.png")
        out.write(image_frame)
    out.release()

def video_map(time=TIME_VIDEO):
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        image_frame = cv2.imread('map/heatmap_0.png')
        width, height, _ = image_frame.shape
        out = cv2.VideoWriter('map/heatmap.avi', fourcc, FRAMES/time, (width,height))

        for i in range(FRAMES):
            image_frame = cv2.imread(f"map/heatmap_{i}.png")
            out.write(image_frame)
        out.release()


if not ONLY_VIDEO:
    env = gym.make('mars_explorer:exploConf-v01', conf=conf)
    observation = env.reset()

    terrain = np.zeros((21, 21))

    for step in range(FRAMES):

        img = env.reset()
        rend = env.render()

        x, y = np.where(env.groundTruthMap == 1.0)
        terrain[y,x] += 1

        if SAVE:
            saveRend(rend)

        plotHeatMap(terrain)

    env.close()

video_env()
video_map()
