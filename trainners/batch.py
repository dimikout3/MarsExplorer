import os
import time
import subprocess as sp
import json

SAVE_LOG = True

if __name__ == "__main__":

    conf = json.load(open('trainner.json','r'))

    for trainner, params in conf["trainners"].items():

        print(f"Running simulation for trainner {trainner}")

        call = ["python",params["path"],
                "--workers", str(params["workers"]),
                "--gamma", str(conf["gamma"]),
                "--steps", str(params["steps"])]

        print(call)

        if SAVE_LOG:
            fout = open(f"{trainner}_report.txt","w")
            s = sp.Popen(call, stdout=fout)
            fout.close()
        else:
            s = sp.Popen(call)

        exitCode = s.wait()
