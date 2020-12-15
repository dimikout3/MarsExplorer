import gym
import numpy as np
from gym.wrappers import Monitor
import time
# from mars_explorer.envs.settings.settings import DEFAULT_CONFIG as conf

# env = gym.make('mars_explorer:explorer-v01', conf=conf)
env = gym.make('mars_explorer:explorer-v01')
# env = gym.make('CartPole-v0')
# env = Monitor(env, './video', force=True)
observation = env.reset()

for _ in range(10):
    action = np.random.randint(4)
    _, _, done, _ = env.step(action)
    if done:
        env.reset()
    env.render()
    time.sleep(1)
    print(done)
    # import pdb; pdb.set_trace()
env.close()
