import numpy as np
import matplotlib.pyplot as plt

# Include the parent direcotry of GEP in python path (not nice looking)
import os, sys
sys.path.append(os.path.join(os.getcwd(), ".."))

from Envs.randomMapGenerator import Generator
from Sensors.lidarSensor import Lidar

import gym


class Grid:

    def __init__(self, size=[30,30], movementCost=0.2, rendering=False):

        self.sizeX = size[0]
        self.sizeY = size[1]

        self.movementCost = movementCost

        self.SIZE = size

        self.rendering = rendering

        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(0.,1.,(self.sizeX, self.sizeY, 1))


    def reset(self, start=[0,0]):

        self.maxSteps = self.sizeX * self.sizeY * 1.5

        # groundTruthMap --> 1.0 obstacle
        #                    0.3 free to move
        gen = Generator(size=[self.sizeX, self.sizeY],
                        number_rows=3, number_columns=3,
                        noise=[0.04,0.04],
                        margins=[0.2, 0.2],
                        obstacle_size=[0.1, 0.1])
        randomMap = gen.get_map().astype(np.double)
        randomMapOriginal = randomMap.copy()
        randomMap[randomMap == 1.0] = 1.0
        randomMap[randomMap == 0.0] = 0.3
        self.groundTruthMap = randomMap

        # for lidar --> 0 free cell
        #               1 obstacle
        self.ldr = Lidar(r=6, channels=32, map=randomMapOriginal)

        obstacles_idx = np.where(self.groundTruthMap == 1.0)
        obstacles_x = obstacles_idx[0]
        obstacles_y = obstacles_idx[1]
        self.obstacles_idx = np.stack((obstacles_x, obstacles_y), axis=1)
        self.obstacles_idx = [list(i) for i in self.obstacles_idx]

        # 0 if not visible/visited, 1 if visible/visited
        self.exploredMap = np.zeros(self.SIZE, dtype=np.double)

        self.x, self.y = start[0], start[1]

        self.state_trajectory = []
        self.reward_trajectory = []

        # starting position is explored
        self._activateLidar()
        self._updateMaps()

        self.outputMap = self.exploredMap.copy()
        self.outputMap[self.x, self.y] = 0.6

        self.new_state = np.reshape(self.outputMap, (self.sizeX, self.sizeY,1))
        self.reward = 0
        self.done = False

        self.timeStep = 0

        return self.new_state


    def action_space_sample(self):
        random = np.random.randint(4)
        return random


    def render(self):

        plt.imshow(self.new_state)
        plt.show()
        plt.close()


    def _choice(self, choice):

        if choice == 0:
            self._move(x=1, y=0)
        elif choice == 1:
            self._move(x=-1, y=0)
        elif choice == 2:
            self._move(x=0, y=1)
        elif choice == 3:
            self._move(x=0, y=-1)


    def _move(self, x, y):

        canditateX = self.x + x
        canditateY = self.y + y

        in_x_axis = canditateX>=0 and canditateX<=(self.sizeX-1)
        in_y_axis = canditateY>=0 and canditateY<=(self.sizeY-1)
        in_obstacles = [canditateX, canditateY] in self.obstacles_idx

        if in_x_axis and in_y_axis and not in_obstacles:
            self.x += x
            self.y += y


    def _updateMaps(self):

        self.pastExploredMap = self.exploredMap.copy()

        lidarX = self.lidarIndexes[:,0]
        lidarY = self.lidarIndexes[:,1]
        self.exploredMap[lidarX, lidarY] = self.groundTruthMap[lidarX, lidarY]

        self.exploredMap[self.x, self.y] = 0.6


    def _activateLidar(self):

        self.ldr.update([self.x, self.y])
        thetas, ranges = self.ldr.thetas, self.ldr.ranges
        indexes = self.ldr.idx

        self.lidarIndexes = indexes


    def _applyRLactions(self,action):

        self._choice(action)
        self._activateLidar()
        self._updateMaps()

        self.outputMap = self.exploredMap.copy()
        self.outputMap[self.x, self.y] = 0.5
        self.new_state = np.reshape(self.outputMap, (self.sizeX, self.sizeY,1))
        self.timeStep += 1


    def _computeReward(self):

        pastExploredCells = np.count_nonzero(self.pastExploredMap)
        currentExploredCells = np.count_nonzero(self.exploredMap)

        # TODO: add fixed cost for moving (-0.5 per move)
        self.reward = currentExploredCells - pastExploredCells - self.movementCost


    def _checkDone(self):

        if self.timeStep > self.maxSteps:
            self.done = True
        elif np.count_nonzero(self.exploredMap) > 0.95*(self.SIZE[0]**2):
            self.done = True
            # self.reward = self.reward + 100
        else:
            self.done = False


    def _updateTrajectory(self):

        self.state_trajectory.append(self.new_state)
        self.reward_trajectory.append(self.reward)


    def step(self, action):

        self._applyRLactions(action)
        self._computeReward()
        self._checkDone()
        self._updateTrajectory()

        info = {}
        return self.new_state, self.reward, self.done, info
