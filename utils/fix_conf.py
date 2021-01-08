import pickle as p
import json
import os

ROOT_PATH = "/home/dkoutras/ray_results/PPO_config_serach"


def update_pkl(dir):

    pkl_dir = f"{dir}/params.pkl"
    obj = p.load(open(pkl_dir,"rb"))

    env_config = obj['env_config']

    if list(env_config.keys()) == ["conf"]:
        return 0

    obj['env_config'] = {}
    obj['env_config']["conf"] = env_config

    p.dump(obj, open(pkl_dir,"wb"))


def update_json(dir):

    json_dir = f"{dir}/params.json"
    obj = json.load(open(json_dir,"r"))

    env_config = obj['env_config']

    if list(env_config.keys()) == ["conf"]:
        return 0

    obj['env_config'] = {}
    obj['env_config']["conf"] = env_config

    json.dump(obj, open(json_dir,"w"))


if __name__ == "__main__":

    runs = os.listdir(ROOT_PATH)

    for run in runs:

        dir = f"{ROOT_PATH}/{run}"

        update_pkl(dir)
        update_json(dir)
