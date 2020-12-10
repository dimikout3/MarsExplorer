import gym
import numpy as np
from gym.wrappers import Monitor
from mars_explorer.envs.settings.settings import DEFAULT_CONFIG as conf

env = gym.make('mars_explorer:explorer-v01', conf=conf)
# env = gym.make('CartPole-v0')
env = Monitor(env, './video', force=True)
observation = env.reset()

for _ in range(600):
    action = np.random.randint(4)
    env.step(action)
    # env.render()
    # import pdb; pdb.set_trace()
env.close()
