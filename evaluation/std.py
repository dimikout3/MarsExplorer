import pandas as pd
import numpy as np
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

from utils import smoothed_compared
from utils import std_factorized
from utils import normalize

import json
import os

plotstyle="ggplot"
plt.style.use(f"{plotstyle}")

SAVE = True
SMOOTH_FACTOR = 100
STD_FACTOR = 0.15
ALPHA = 0.25

HUMAN_LEVEL = 0.9
HUMAN_STD = 0.0
HUMAN_C = 'g'


def combined():

    compared_folder = os.path.join(os.getcwd(), "compared")
    try:
        os.makedirs(compared_folder)
    except OSError:
        if not os.path.isdir(compared_folder):
            raise

    for agent in df.Agent.unique():
        print(f"Ploting Combined plot for agent {agent}")
        sns.lineplot(
                     # data=df[df.iteration<1000*10000/1_000_000],
                     data=df[df.Agent == agent],
                     x="iteration",
                     y="mean",
                     dashes=False,
                     label=agent)

        x_axis = df[ df.Agent == agent]['iteration']
        y_axis = df[ df.Agent == agent]['mean']
        std = df[ df.Agent == agent]['std']
        lower = y_axis - std
        upper = y_axis + std
        plt.fill_between(x_axis, lower, upper, alpha=ALPHA)

    plt.hlines( y = HUMAN_LEVEL,
                xmin = 1.8,
                xmax = df.iteration.max(),
                label="human",
                color=HUMAN_C,
                linestyle='--')
    if HUMAN_STD>0.:
        plt.fill_between(df.iteration,
                         HUMAN_LEVEL-HUMAN_STD,
                         HUMAN_LEVEL+HUMAN_STD,
                         alpha=ALPHA,
                         color=HUMAN_C)

    plt.ylim(0,1.)

    plt.ylabel("Score normalized")
    plt.xlabel("Timesteps (M)")
    plt.legend(loc='upper left')


    if SAVE:
        plt.savefig(f"{compared_folder}/Compared_std.png", dpi=300)
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":

    ref = {"PPO":"/home/dkoutras/ray_results/Agent_search/result_PPO.json",
           "SAC":"/home/dkoutras/ray_results/Agent_search/result_SAC.json",
           "Rainbow":"/home/dkoutras/ray_results/Agent_search/result_DQN.json",
           "A3C":"/home/dkoutras/ray_results/Agent_search/result_A2C.json"}

    df = pd.DataFrame()

    data = {}
    data["Agent"] = []
    data["mean"] = []
    data["std"] = []
    data["iteration"] = []

    for agent, json_file in ref.items():

        result = []
        with open(json_file) as f:
            for line in f:
                result.append(json.loads(line))

        min = -400
        max = 800

        score = []
        for iteration, result_line in enumerate(result):
            data["Agent"].append(agent)
            normalized_mean = (result_line['episode_reward_mean']-min)/(max-min)
            data["mean"].append(normalized_mean)

            episodes_rewards = result_line['hist_stats']['episode_reward']
            normalized_std = (np.std(episodes_rewards)-min)/(max-min)
            if STD_FACTOR<1:
                normalized_std = std_factorized(normalized_std, STD_FACTOR)

            data["std"].append(normalized_std)
            data["iteration"].append(iteration)

    df = pd.DataFrame(data)
    df['iteration'] = df['iteration']*4000/1_000_000

    df = normalize(df)
    df = smoothed_compared(df, SMOOTH_FACTOR, ref)

    df = df[df.iteration<900*10000/1_000_000]

    combined()
