import argparse
import time

import gym as gym
import numpy as np
import collections
import matplotlib.pyplot as plt
import pickle as p

import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter

from Envs.gridEnvPixelsFull import Grid as grid
from Agents.agent import Agent
from Buffers.experience import ExperienceBuffer

from Models.dqn_model import DQN, DQN_simple

EPISODES = 5000

GAMMA = 0.95
BATCH_SIZE = 32
REPLAY_SIZE = 30000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = REPLAY_SIZE

EPSILON_DECAY_LAST_FRAME = int(EPISODES*150*0.7)
EPSILON_START = 0.4
EPSILON_FINAL = 0.01

MEAN_REWARD_BOUND = 20000
SHOW_EVERY = 100

BEST_M_REWARD = 475

def calc_loss(batch, net, tgt_net, device="cpu"):
    states, actions, rewards, dones, next_states = batch
    states_v = torch.tensor(np.array(
        states, copy=False)).to(device)
    next_states_v = torch.tensor(np.array(
        next_states, copy=False)).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    state_action_values = net(states_v).gather(
        1, actions_v.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        next_state_values = tgt_net(next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0
        next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * GAMMA + \
                                   rewards_v
    return nn.MSELoss()(state_action_values,
                        expected_state_action_values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False,
                        action="store_true", help="Enable cuda")
    parser.add_argument("-m", "--model", default="dqn_model_default.dat",
                        help="Model file to load")
    parser.add_argument("-o", "--output", default="dqn_model_out.dat",
                        help="Model file to be saved")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    env = gym.make('CartPole-v0') #CartPole-v1, MountainCar-v0
    action_dim = env.action_space.n
    observation_dim = env.observation_space.shape

    net = DQN_simple(observation_dim ,action_dim).to(device)
    tgt_net = DQN_simple(observation_dim ,action_dim).to(device)
    writer = SummaryWriter(comment="DQN-GEP")
    print(net)
    if "default" not in args.model:
        print(f"Updating existing NN model {args.model}")
        net.load_state_dict(torch.load(args.model))
        tgt_net.load_state_dict(torch.load(args.model))

    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_m_reward = None
    episode = 0
    initial_time = time.time()
    best_m_reward = BEST_M_REWARD

    loss_t = None

    while episode<EPISODES:
        frame_idx += 1
        epsilon = max(EPSILON_FINAL, EPSILON_START -
                      frame_idx / EPSILON_DECAY_LAST_FRAME)

        reward = agent.play_step(net, epsilon, device=device)
        if reward is not None:

            episode += 1

            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            m_reward = np.mean(total_rewards[-100:])

            print(f"{frame_idx}: done {len(total_rewards)} games, ma {m_reward:.2f}, eps {epsilon:.2f}, speed {speed:.2f} f/s, elapsed time {time.time() - initial_time:.2f}[sec]")

            writer.add_scalar("reward_100", m_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)
            writer.add_scalar("epsilon", epsilon, frame_idx)

            if loss_t != None:
                writer.add_scalar("loss", loss_t, frame_idx)

            if best_m_reward is None or best_m_reward < m_reward:
                torch.save(net.state_dict(), args.output)
                if best_m_reward is not None:
                    print("Best reward updated %.3f -> %.3f" % (
                        best_m_reward, m_reward))
                best_m_reward = m_reward

            if m_reward > MEAN_REWARD_BOUND:
                print("Solved in %d frames!" % frame_idx)
                break

        if len(buffer) < REPLAY_START_SIZE:
            continue

        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())

        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch, net, tgt_net, device=device)
        loss_t.backward()
        optimizer.step()

    print(f"\nSaving torch NN  at dir:{args.output}")
    torch.save(net.state_dict(), args.output)

    writer.close()
