# from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
from ray.rllib.agents.sac import SACTrainer, DEFAULT_CONFIG
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print
import ray

import json
import time

from mars_explorer.envs.explorer import Explorer

from tensorboardX import SummaryWriter

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
    config['gamma'] = 0.3

    config['monitor'] = False

    # PPO config ...
    # config['lr'] = 1e-4
    # config['train_batch_size']
    config['model']['dim'] = 21
    # config['model']['conv_filters'] = [ [16, [3, 3], 1],
    #                                     # [16, [3, 3], 1],
    #                                     # [32, [config['model']['dim'], config['model']['dim']], 1]]#,
    #                                     [32, [config['model']['dim'], config['model']['dim']], 1]]
    config['model']['conv_filters'] = [ [8, [4, 4], 4],
                                        [16, [2, 2], 2],
                                        [512, [3, 3], 1]]#,
    # trainner = PPOTrainer(config=config, env="custom-explorer")

    config["buffer_size"] = 500_000
    # If True prioritized replay buffer will be used.
    config["prioritized_replay"] = True
    trainner = SACTrainer(config=config, env="mars_explorer:explorer-v01")

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
