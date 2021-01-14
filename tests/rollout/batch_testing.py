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

        level_dir = os.path.join(os.getcwd(), f"{run}")
        try:
            os.makedirs(level_dir)
        except OSError:
            if not os.path.isdir(level_dir):
                raise

        checkpoint_list = os.listdir(f"{ROOT_PATH}/{run}")
        checkpoint_list = [filename for filename in checkpoint_list if filename.startswith("checkpoint")]

        for checkpoint in checkpoint_list:

            checkpoint_dir = os.path.join(ROOT_PATH, f"{run}/{checkpoint}")
            try:
                os.makedirs(checkpoint_dir)
            except OSError:
                if not os.path.isdir(checkpoint_dir):
                    raise

            checkpoint_output_dir = os.path.join(os.getcwd(), f"{run}/{checkpoint}")
            try:
                os.makedirs(checkpoint_output_dir)
            except OSError:
                if not os.path.isdir(checkpoint_output_dir):
                    raise

            checkpoint_tune = checkpoint.replace("_","-")

            call = ["python","rollout.py",
                    f"{checkpoint_dir}/{checkpoint_tune}",
                    "--run", "PPO",
                    "--env", "mars_explorer:exploConf-v01",
                    "--episodes", "40",
                    "--save-info",
                    "--video-dir", checkpoint_output_dir,
                    "--out", f"{checkpoint_output_dir}/rollout_{run}.pkl"]

            print(str(call))

            if SAVE_LOG:
                fout = open(f"{run}_report.txt","w")
                s = sp.Popen(call, stdout=fout)
                fout.close()
            else:
                s = sp.Popen(call)

            exitCode = s.wait()
