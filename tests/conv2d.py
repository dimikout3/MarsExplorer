from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
# from ray.rllib.agents.sac import SACTrainer, DEFAULT_CONFIG
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print
import ray

import json
import time

from mars_explorer.envs.explorer import Explorer

def env_creator(env_config):
    return Explorer()

ray.init(num_gpus=1)
register_env("custom-explorer", env_creator)

config = DEFAULT_CONFIG.copy()
config['num_workers'] = 1
config['num_gpus'] = 1
config['framework'] = "torch"
config['gamma'] = 0.1

config['monitor'] = False

# PPO config ...
# config['lr'] = 1e-4
# config['train_batch_size']
config['model']['dim'] = 21
config['model']['conv_filters'] = [ [8, [4, 4], 2],
                                    [16, [2, 2], 2],
                                    [512, [6, 6], 1]]#,
                                    #[config['train_batch_size'], 4, 1, 1]]


# trainner = PPOTrainer(config=config, env="mars_explorer:explorer-v01")
trainner = PPOTrainer(config=config, env="custom-explorer")
# import pdb; pdb.set_trace()

PATH = "/home/dkoutras/ray_results/290_out_of_400/checkpoint_2991/checkpoint-2991"
trainner.restore(PATH)
import pdb; pdb.set_trace()

for _ in range(10):
    initial_time = time.time()
    result = trainner.train()
    print(f"mean:{result['episode_reward_mean']} time:{time.time() - initial_time:.2f}[sec]")
