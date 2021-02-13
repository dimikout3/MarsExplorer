import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import json
import os
import pickle as p

plotstyle="ggplot"
plt.style.use(f"{plotstyle}")

SAVE=False

LEVELS = [
"1_1_1", "1_2_1", "1_1_2", "1_2_2",
"2_1_1", "2_2_1", "2_1_2", "2_2_2",
"3_1_1", "3_2_1", "3_1_2", "3_2_2"]


def violin():

    sns.violinplot(data = df,
                   x = "level",
                   y = "mean")

    plt.ylim(0.4,.8)

    if SAVE:
        plt.savefig(f"{step}_heatmap.png", bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def swarm():

    g = sns.catplot(x="level", y="mean",
                    kind="violin",
                    inner=None,
                    data=df)
    sns.swarmplot(x="level", y="mean",
                  color="k",
                  size=.8,
                  data=df,
                  ax=g.ax)

    # plt.ylim(0.4,.8)

    if SAVE:
        plt.savefig(f"{step}_heatmap.png", bbox_inches='tight')
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":

    df = pd.DataFrame()

    data = {}
    data["level"] = []
    data["mean"] = []

    for level in LEVELS:
        path = f"spatio_temporal/rollout_{level}.pkl"
        rollouts = p.load(open(path,'rb'))

        for rollout in rollouts:

            episodic_reward = []

            for time_i, time_step in enumerate(rollout):
                episodic_reward.append(rollout[time_i][3])

            if time_i < 30: continue
            data["level"].append(level)
            data["mean"].append(np.mean(episodic_reward))

    df = pd.DataFrame(data)
    # violin()
    swarm()
