from ray.rllib.trainners.ppo import PPOTrainer, DEFAULT_CONFIG
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print
import ray

import json
import time

from Envs.lidar_V02 import Grid as grid


def env_creator(env_config):
    return grid(size=[42,42])

if __name__ == "__main__":

    ray.init()
    register_env("my_env", env_creator)

    config = DEFAULT_CONFIG.copy()
    config['num_workers'] = 5
    # config['num_envs_per_worker'] = 1
    # config['num_cpus_per_worker'] = 1
    config['framework'] = "torch"

    trainner = PPOTrainer(config=config, env="my_env")

    N = 1000
    results = []
    episode_data = []
    episode_json = []

    for n in range(N):

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
