from ray.rllib.agents.dqn import DQNTrainer, DEFAULT_CONFIG
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print
import ray

import json
import time

from mars_explorer.envs.explorer import Explorer

from tensorboardX import SummaryWriter

PATH = "/home/dkoutras/ray_results/DQN_mars_explorer:explorer-v01_2020-12-15_02-07-087so6r6r9/checkpoint_751/checkpoint-751"

def env_creator(env_config):
    return Explorer()

if __name__ == "__main__":

    ray.init(num_gpus=1)
    register_env("custom-explorer", env_creator)

    config = DEFAULT_CONFIG.copy()
    config['num_workers'] = 8
    config['framework'] = "torch"
    config['gamma'] = 0.9

    # NN vision
    config['model']['dim'] = 21
    config['model']["fcnet_hiddens"] = [256,256]
    config['model']['conv_filters'] = [ [8, [4, 4], 2],
                                        [16, [2, 2], 2],
                                        [512, [6, 6], 1]]#,
    #  DQN config
    config['v_min'] = -15
    config['v_max'] = 30
    config['noisy'] = False
    trainner = DQNTrainer(config=config, env="mars_explorer:explorer-v01")

    if PATH != "":
        print(f"Loading model {PATH}")
        trainner.restore(PATH)
    else:
        print(f"Starting without any a priori knowledge")
    N_start = 0
    N_finish = 30000
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
