from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print
import ray

import json
import time

from mars_explorer.envs.explorer import Explorer
from mars_explorer.envs.settings.settings import DEFAULT_CONFIG as env_config

def env_creator(env_config):
    return Explorer(env_config)

if __name__ == "__main__":

    ray.init(num_gpus=1)
    register_env("custom-explorer", env_creator)

    config = DEFAULT_CONFIG.copy()
    config['num_workers'] = 8
    config['num_gpus'] = 1
    config['framework'] = "torch"
    config['gamma'] = 0.9

    config['monitor'] = False

    config["env_config"] = env_config
    config["env_config"]["viewer"]["drone_img"] = "render/images/drone.png"
    config["env_config"]["viewer"]["obstacle_img"] = "render/images/block.png"
    config["env_config"]["viewer"]["background_img"] = "render/images/mars.jpg"
    config["env_config"]["viewer"]["light_mask"] = "render/images/light_350_hard.png"
    trainner = PPOTrainer(config=config, env="custom-explorer")

    N = 1000
    results = []
    episode_data = []
    episode_json = []

    for n in range(0, N):

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
