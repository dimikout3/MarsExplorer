from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo import DEFAULT_CONFIG as PPO_CONFIG

from mars_explorer.envs.settings import DEFAULT_CONFIG as DEFAULT_ENV_CONFIG

import json

def defaultConfig(args):

    if args.agent == "PPO":
        config = PPO_CONFIG.copy()

    config['num_workers'] = args.workers
    config['num_gpus'] = 1
    config['framework'] = "torch"
    config['gamma'] = args.gamma

    config['model']['dim'] = 21
    config['model']['conv_filters'] = [ [8, [3, 3], 2],
                                        [16, [2, 2], 2],
                                        [512, [6, 6], 1]]

    return config

def agentConfig(config, args):
    conf_json = json.load(open(args.configuration,'r'))
    for param, value in conf_json["run"][args.run]["agent_conf"].items():
        config[param] = value
    return config


def envConfig(config, args):
    conf_json = json.load(open(args.configuration,'r'))
    env_config = DEFAULT_ENV_CONFIG
    for param, value in conf_json["run"][args.run]["env_conf"].items():
        env_config[param] = value
    return env_config


def getConfig(args):

    config = defaultConfig(args)
    config = agentConfig(config, args)
    config['env_config'] = envConfig(config, args)

    return config


def getTrainner(args):

    config = getConfig(args)

    if args.agent == "PPO":
        trainner = PPOTrainer(config=config, env="custom-explorer")

    return trainner
