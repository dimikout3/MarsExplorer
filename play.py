import time
import argparse
import numpy as np

import torch

from Envs.gridEnvPixelsFull import Grid as grid
from Agents.agent import Agent
from Buffers.experience import ExperienceBuffer

from Models.dqn_model import DQN

import collections

DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
FPS = 25

SIZE_X = 10
SIZE_Y = 10
INPUT_CHANNELS = 1
INPUT_SHAPE = (INPUT_CHANNELS, SIZE_X, SIZE_Y)
ACTION_SHAPE = 4

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True,
                        help="Model file to load")
    parser.add_argument("-e", "--env", default=DEFAULT_ENV_NAME,
                        help="Environment name to use, default=" +
                             DEFAULT_ENV_NAME)
    parser.add_argument("-r", "--record", help="Directory for video")
    parser.add_argument("--no-vis", default=True, dest='vis',
                        help="Disable visualization",
                        action='store_false')
    args = parser.parse_args()

    env = grid(size=[SIZE_X, SIZE_Y], renderingThreshold=44)

    net = DQN(INPUT_SHAPE ,ACTION_SHAPE)
    state = torch.load(args.model, map_location=lambda stg, _: stg)
    net.load_state_dict(state)

    state = env.reset()
    total_reward = 0.0
    c = collections.Counter()

    while True:
        start_ts = time.time()
        if args.vis:
            env.render()
        state_v = torch.tensor(np.array([state], copy=False))
        q_vals = net(state_v).data.numpy()[0]
        action = np.argmax(q_vals)
        c[action] += 1
        state, reward, done = env.step(action)
        total_reward += reward
        if done:
            break

    print(f"Game finished after {time.time() - start_ts}[sec]")
    print(f"Total reward: {total_reward}")
    print("Action counts:", c)
    print(f"Inspect the frames at dir:{args.record}")
