from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print
import ray

import json
import time

from mars_explorer.envs.explorer import Explorer

# PATH = "/home/dkoutras/ray_results/PPO_custom-explorer_2020-12-11_00-08-54v0rb3cxa/checkpoint_1831/checkpoint-1831"
PATH = ""

def env_creator(env_config):
    return Explorer()

if __name__ == "__main__":

    ray.init(num_gpus=1)
    register_env("custom-explorer", env_creator)

    config = DEFAULT_CONFIG.copy()
    config['num_workers'] = 8
    config['num_gpus'] = 1
    config['framework'] = "torch"
    config['gamma'] = 0.9

    config['monitor'] = False

    trainner = PPOTrainer(config=config, env="custom-explorer")

    if PATH != "":
        print(f"\nLoading trainner from dir {PATH}")
        trainner.restore(PATH)
    else:
        print(f"Starting trainning without a priori knowledge")

    N_start = 0
    N_finish = 3000
    results = []
    episode_data = []
    episode_json = []

    for n in range(N_start, N_finish):

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

        if n % 10 == 0:
            checkpoint = trainner.save()
            print("checkpoint saved at", checkpoint)

        print(f'{n:3d}: Min/Mean/Max reward: {result["episode_reward_min"]:8.4f}/{result["episode_reward_mean"]:8.4f}/{result["episode_reward_max"]:8.4f} time:{time.time() - initial_time:.2f}[sec]')
