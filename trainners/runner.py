from ray.tune.registry import register_env
import ray

from utils import getTrainner

import json
import time
import argparse

from mars_explorer.envs.explorerConf import ExplorerConf

from tensorboardX import SummaryWriter

PATH = ""

def env_creator(env_config):
    return ExplorerConf(env_config)

def getArgs():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '-w', '--workers',
        default=4,
        type=int,
        help='Number of worker (=cores)')
    argparser.add_argument(
        '-g', '--gamma',
        default=0.9,
        type=float,
        help='Gamma to be used in Q calculations')
    argparser.add_argument(
        '-s', '--steps',
        default=3000,
        type=int,
        help='Number of steps the algorithm will run')
    argparser.add_argument(
        '-a', '--agent',
        default="PPO",
        type=str,
        help='Agent to be used')
    argparser.add_argument(
        '-r', '--run',
        default="No-Provided",
        type=str,
        help='Specific run identifier')
    argparser.add_argument(
        '-c', '--configuration',
        default="trainner.json",
        type=str,
        help='General configuration for the specified trainner')
    return argparser.parse_args()


if __name__ == "__main__":

    args = getArgs()

    ray.init(num_gpus=1)
    register_env("custom-explorer", env_creator)

    trainner = getTrainner(args)

    if PATH != "":
        print(f"\nLoading trainner from dir {PATH}")
        trainner.restore(PATH)
    else:
        print(f"Starting trainning without a priori knowledge")

    N_start = 0
    N_finish = args.steps
    results = []
    episode_data = []
    episode_json = []

    writer = SummaryWriter(comment="PPO-GEP")

    for batch in range(N_start, N_finish):

        initial_time = time.time()

        result = trainner.train()
        results.append(result)

        episode = {'n': batch,
                   'episode_reward_min':  result['episode_reward_min'],
                   'episode_reward_mean': result['episode_reward_mean'],
                   'episode_reward_max':  result['episode_reward_max'],
                   'episode_len_mean':    result['episode_len_mean']}

        episode_data.append(episode)
        episode_json.append(json.dumps(episode))

        writer.add_scalar("reward_min", result['episode_reward_min'], batch)
        writer.add_scalar("reward_mean", result['episode_reward_mean'], batch)
        writer.add_scalar("reward_max", result['episode_reward_max'], batch)

        if batch % 10 == 0:
            checkpoint = trainner.save()
            print("checkpoint saved at", checkpoint)

        print(f'{batch:3d}: Min/Mean/Max reward: {result["episode_reward_min"]:8.4f}/{result["episode_reward_mean"]:8.4f}/{result["episode_reward_max"]:8.4f} time:{time.time() - initial_time:.2f}[sec]')

    writer.close()
    print("\n Finished successfully")
