import gym
import numpy as np
from gym.wrappers import Monitor

env = gym.make('mars_explorer:explorer-v01')
# env = gym.make('CartPole-v0')
env = Monitor(env, './video', force=True)
observation = env.reset(start=[17,17])

for _ in range(300):
    action = np.random.randint(4)
    env.step(action)
    # env.render()
    # import pdb; pdb.set_trace()
env.close()
