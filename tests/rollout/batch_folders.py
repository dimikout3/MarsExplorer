import pandas as pd
import numpy as np

import os
import time
import subprocess as sp
import json

ROOT_PATH = "/home/dkoutras/ray_results/PPO_config_serach"
SAVE_LOG = False


if __name__ == "__main__":

    runs = os.listdir(ROOT_PATH)

    for run in runs:

        result_dir = os.path.join(os.getcwd(), f"{run}")
        try:
            os.makedirs(result_dir)
        except OSError:
            if not os.path.isdir(result_dir):
                raise

        call = ["python","rollout.py",
                f"{ROOT_PATH}/{run}/checkpoint_191/checkpoint-191",
                "--run", "PPO",
                "--env", "mars_explorer:exploConf-v01",
                "--episodes", "40",
                "--save-info",
                "--video-dir", result_dir,
                "--out", f"{result_dir}/rollout_{run}.pkl"]

        print(str(call))

        if SAVE_LOG:
            fout = open(f"{run}_report.txt","w")
            s = sp.Popen(call, stdout=fout)
            fout.close()
        else:
            s = sp.Popen(call)

        exitCode = s.wait()
