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

    if SAVE:
        plt.savefig(f"{step}_heatmap.png", bbox_inches='tight')
    else:
        plt.show()
    plt.close()



if __name__ == "__main__":

    df = p.load(open('DataFrame.pkl','rb'))
    df = df[df.iteration > 0.9*df.iteration.max()]
    df["mean"] = df["mean"]*0.8

    violin()
