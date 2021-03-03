import numpy as np
import time
import json
import pickle as p
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


plotstyle="ggplot"
plt.style.use(f"{plotstyle}")


CASES = [["utility_42x42.p", "Area 42x42", "Utility [8]"],
         ["cost_42x42.p", "Area 42x42", "Cost [8]"],
         ["ppo_42x42.p", "Area 42x42", "PPO [22]"],
         ["utility_84x84.p", "Area 84x84", "Utility [8]"],
         ["cost_84x84.p", "Area 84x84", "Cost [8]"],
         ["ppo_84x84.p", "Area 84x84", "PPO [22]"]]


if __name__ == "__main__":

    dict = {}
    dict["Terrain"] = []
    dict["Distance Covered [m]"] = []
    dict["Eplored Area [%]"] = []
    dict["Exploration Policy"] = []

    for file, area, exploration_type in CASES:

        data = p.load(open(file,"rb"))

        for game in data:
            for distance,explored in enumerate(game):
                dict["Terrain"].append(area)
                dict["Distance Covered [m]"].append(distance)
                dict["Eplored Area [%]"].append(explored)
                dict["Exploration Policy"].append(exploration_type)

    df = pd.DataFrame(dict)

    sns.lineplot(data=df, x="Distance Covered [m]", y="Eplored Area [%]",
                 hue="Exploration Policy", style="Terrain")
    plt.show()
