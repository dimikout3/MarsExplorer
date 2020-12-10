from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print
import ray

import json
import time

from Envs.lidar_V02 import Grid as grid

PATH = "/home/dkoutras/ray_results/PPO_lidar-V02_2020-11-27_19-23-08/checkpoint_981/checkpoint-981"

def env_creator(env_config):
    return grid(size=[42,42])

if __name__ == "__main__":

    ray.init(num_gpus=1)
    register_env("lidar-V02", env_creator)

    config = DEFAULT_CONFIG.copy()
    config['num_workers'] = 8
    config['num_gpus'] = 1
    config['framework'] = "torch"
    config['gamma'] = 0.9

    trainner = PPOTrainer(config=config, env="lidar-V02")

    print(f"\nLoading trainner from dir {PATH}")
    trainner.restore(PATH)

    N = 2000
    results = []
    episode_data = []
    episode_json = []

    for n in range(1000, 1000+N):

        initial_time = time.time()

        result = trainner.train()
        results.append(result)

        episode = {'n': n,
                   'episode_reward_min':  result['episode_reward_min'],
                   'episode_reward_mean': result['episode_reward_mean'],
                   'episode_reward_max':  result['episode_reward_max'],
                   'episode_len_mean':    result['episode_len_mean']}

        episode_data.append(episode)
        episode_json.append(json.dumps(episode))

        if n % 20 == 0:
            checkpoint = trainner.save()
            print("checkpoint saved at", checkpoint)

        print(f'{n:3d}: Min/Mean/Max reward: {result["episode_reward_min"]:8.4f}/{result["episode_reward_mean"]:8.4f}/{result["episode_reward_max"]:8.4f} time:{time.time() - initial_time:.2f}[sec]')
