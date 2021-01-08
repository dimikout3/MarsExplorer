import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import json
import os

from utils import smoothed
from utils import reject_outliers
from utils import std_factorized

plotstyle="ggplot"
plt.style.use(f"{plotstyle}")

SAVE = True
SMOOTH_FACTOR = 5
OUTLIERS_FACTOR = 4
STD_FACTOR = 0.1
ALPHA = 0.25

def get_level():
    lvl = "lvl-["
    for core_capability in config["params"]:
        for axes_level, value in config["params"][core_capability].items():
            if env_config[core_capability] == value:
                lvl = lvl+axes_level+","
    lvl = lvl[0:-1]
    lvl += "]"
    return lvl


def standalone():
    print("Ploting Standalone plots")
    levels = df.level.unique()

    # generate_standaone_folder
    standalone_folder = os.path.join(os.getcwd(), "standalone")
    try:
        os.makedirs(standalone_folder)
    except OSError:
        if not os.path.isdir(standalone_folder):
            raise

    for level in levels:
        sns.lineplot(data = df[ df.level == level],
                     x='iteration',
                     y='mean')

        x_axis = df[ df.level == level]['iteration']
        y_axis = df[ df.level == level]['mean']
        std = df[ df.level == level]['std']
        lower = y_axis - std
        upper = y_axis + std
        plt.fill_between(x_axis, lower, upper, alpha=0.3)

        plt.ylabel("Score normalized")
        plt.xlabel("Timesteps (M)")
        plt.title(level)

        if SAVE:
            plt.savefig(f"{standalone_folder}/{level}.png", dpi=300)
        else:
            plt.show()
        plt.close()


def categorized():
    print("Ploting Categorized plots")

    categorized_folder = os.path.join(os.getcwd(), "categorized")
    try:
        os.makedirs(categorized_folder)
    except OSError:
        if not os.path.isdir(categorized_folder):
            raise

    for title, levels in config["categorized"].items():
        df_temp = pd.DataFrame()
        for level in levels:
            df_temp = pd.concat([df_temp, df[df.level == level]])

        sns.lineplot(data=df_temp,
                     x="iteration",
                     y="mean",
                     hue='level',
                     dashes=False)

        plt.ylabel("Score normalized")
        plt.xlabel("Timesteps (M)")
        plt.title(title)

        if SAVE:
            plt.savefig(f"{categorized_folder}/{title}.png", dpi=300)
        else:
            plt.show()
        plt.close()


def categorized_std():
    print("Ploting Categorized plots (std included)")

    categorized_folder = os.path.join(os.getcwd(), "categorized_std")
    try:
        os.makedirs(categorized_folder)
    except OSError:
        if not os.path.isdir(categorized_folder):
            raise

    for title, levels in config["categorized"].items():

        for level in levels:

            sns.lineplot(data=df[df.level == level],
                         x="iteration",
                         y="mean",
                         label=level,
                         dashes=False)
            x_axis = df[df.level == level]['iteration']
            y_axis = df[df.level == level]['mean']
            std = df[df.level == level]['std']
            lower = y_axis - std
            upper = y_axis + std
            plt.fill_between(x_axis, lower, upper, alpha=ALPHA)

        plt.ylabel("Score normalized")
        plt.xlabel("Timesteps (M)")
        plt.title(title)

        if SAVE:
            plt.savefig(f"{categorized_folder}/{title}.png", dpi=300)
        else:
            plt.show()
        plt.close()


def combined():
    print("Ploting Combined plot")

    combined_folder = os.path.join(os.getcwd(), "combined")
    try:
        os.makedirs(combined_folder)
    except OSError:
        if not os.path.isdir(combined_folder):
            raise

    sns.lineplot(data=df,
                 x="iteration",
                 y="mean",
                 hue='level',
                 dashes=False)
    plt.ylabel("Score normalized")
    plt.xlabel("Timesteps (M)")
    plt.title("Combined")

    if SAVE:
        plt.savefig(f"{combined_folder}/Combined.jpg", dpi=300)
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":

    config = json.load(open("evaluate.json","r"))
    runs = os.listdir(config["root"])

    df = pd.DataFrame()

    data = {}
    data["level"] = []
    data["mean"] = []
    data["std"] = []
    data["iteration"] = []

    for run in runs:

        result = []
        with open(f"{config['root']}/{run}/result.json") as f:
            for line in f:
                result.append(json.loads(line))

        params_dir = f"{config['root']}/{run}/params.json"
        params = json.load(open(params_dir,'r'))

        env_config = params["env_config"]["conf"]
        min = -env_config["bonus_reward"]
        max = env_config["bonus_reward"] + env_config["size"][0]*env_config["size"][1]

        lvl = get_level()

        score = []
        for iteration, result_line in enumerate(result):
            data["level"].append(lvl)
            normalized_mean = (result_line['episode_reward_mean']-min)/(max-min)
            data["mean"].append(normalized_mean)

            episodes_rewards = result_line['hist_stats']['episode_reward']
            if OUTLIERS_FACTOR<3:
                episodes_rewards = reject_outliers(episodes_rewards,
                                                   m=OUTLIERS_FACTOR)

            normalized_std = (np.std(episodes_rewards)-min)/(max-min)
            if STD_FACTOR<1:
                normalized_std = std_factorized(normalized_std, STD_FACTOR)

            data["std"].append(normalized_std)
            data["iteration"].append(iteration)

    df = pd.DataFrame(data)
    df["iteration"] = df["iteration"]*4000/1_000_000

    df = smoothed(df, SMOOTH_FACTOR)

    df = df[df.iteration<0.97*df.iteration.max()]

    standalone()
    categorized()
    categorized_std()
    combined()
