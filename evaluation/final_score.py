import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import json
import os
import pickle as p

plotstyle="ggplot"
plt.style.use(f"{plotstyle}")

PATH = "rollouts.pkl"
SAVE = False


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
                  size=1.2,
                  data=df,
                  ax=g.ax)

    # plt.ylim(0.4,.8)

    if SAVE:
        plt.savefig(f"{step}_heatmap.png", bbox_inches='tight')
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":

    # df = p.load(open('DataFrameRaw.pkl','rb'))
    df = p.load(open('DataFrame.pkl','rb'))
    df = df[df.iteration > 0.8*df.iteration.max()]
    df["mean"] = df["mean"]*0.8

    # violin()
    swarm()
