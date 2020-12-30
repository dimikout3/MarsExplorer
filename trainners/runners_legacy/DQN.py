from ray.rllib.agents.dqn import DQNTrainer, DEFAULT_CONFIG
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print
import ray

import json
import time
import argparse

from mars_explorer.envs.explorer import Explorer

from tensorboardX import SummaryWriter

PATH = ""

def env_creator(env_config):
    return Explorer()

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
    return argparser.parse_args()


if __name__ == "__main__":

    args = getArgs()

    ray.init(num_gpus=1)
    register_env("custom-explorer", env_creator)

    config = DEFAULT_CONFIG.copy()
    config['num_workers'] = args.workers
    config['framework'] = "torch"
    config['gamma'] = args.gamma

    # NN vision
    config['model']['dim'] = 21
    config['model']['conv_filters'] = [ [8, [3, 3], 2],
                                        [16, [2, 2], 2],
                                        [512, [6, 6], 1]]
    #  DQN config
    config['v_min'] = -400
    config['v_max'] = 400
    config['noisy'] = False
    trainner = DQNTrainer(config=config, env="mars_explorer:explorer-v01")

    if PATH != "":
        print(f"Loading model {PATH}")
        trainner.restore(PATH)
    else:
        print(f"Starting without any a priori knowledge")
    N_start = 0
    N_finish = args.steps
    results = []
    episode_data = []
    episode_json = []

    writer = SummaryWriter(comment="SAC-GEP")

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
