import os
import time
import subprocess as sp
import json

SAVE_LOG = False

if __name__ == "__main__":

    conf_path = "trainnerV2.json"
    conf = json.load(open(conf_path,'r'))

    for run, params in conf["run"].items():

        print(f"Running simulation for run {run}")

        call = ["python","runner.py",
                "--agent", str(params["agent"]),
                "--run", str(run),
                "--workers", str(params["workers"]),
                "--gamma", str(conf["gamma"]),
                "--configuration", str(conf_path),
                "--steps", str(params["steps"])]

        print(str(call))

        if SAVE_LOG:
            fout = open(f"{run}_report.txt","w")
            s = sp.Popen(call, stdout=fout)
            fout.close()
        else:
            s = sp.Popen(call)

        exitCode = s.wait()
