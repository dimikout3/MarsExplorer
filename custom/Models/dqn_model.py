import torch
import torch.nn as nn
import numpy as np

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=2, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x.float()).view(x.size()[0], -1)
        return self.fc(conv_out)

class DQN_simple(nn.Module):


    def __init__(self, input_shape, n_actions, h_size = 64):
        super(DQN_simple, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_shape[0], h_size),
            nn.ReLU(),
            nn.Linear(h_size, h_size),
            nn.ReLU(),
            nn.Linear(h_size, n_actions)
        )

    def forward(self, x):
        return self.fc(x.float())


class DQN_ORIGINAL(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x.float()).view(x.size()[0], -1)
        return self.fc(conv_out)


class DQN_ACTION(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        # import pdb; pdb.set_trace()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 8, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=2, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc1 = nn.Sequential(
            nn.Linear(conv_out_size, 400),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(400 + n_actions, 200),
            nn.ReLU(),
            nn.Linear(200, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, state):

        '''State --> (ExploredMap, (x,y))'''

        obs, actions = state
        conv_out = self.conv(obs).view(obs.size()[0], -1)
        fc1_out = self.fc1(conv_out)
        return self.fc2(fc1_out+actions)
