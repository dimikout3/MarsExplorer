import numpy as np

from mars_explorer.utils.randomMapGenerator import Generator
from mars_explorer.utils.lidarSensor import Lidar
from mars_explorer.render.viewer import Viewer

import gym
from gym import error, spaces, utils
from gym.utils import seeding

class Explorer(gym.Env):
    metadata = {'render.modes': ['rgb_array'],
                'video.frames_per_second': 3}
    # def __init__(self, size=[42,42], movementCost=0.2):
    def __init__(self, conf):

        self.conf = conf

        self.sizeX = conf["size"][0]
        self.sizeY = conf["size"][1]

        self.movementCost = conf["movementCost"]

        self.SIZE = conf["size"]

        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(0.,1.,(self.sizeX, self.sizeY, 1))

        self.viewerActive = False

    # def reset(self, start=[0,0]):
    def reset(self):

        self.maxSteps = self.sizeX * self.sizeY * 1.5

        # groundTruthMap --> 1.0 obstacle
        #                    0.3 free to move
        #                    0.0 unexplored
        #                    0.6 robot
        gen = Generator(self.conf)
        randomMap = gen.get_map().astype(np.double)
        randomMapOriginal = randomMap.copy()
        randomMap[randomMap == 1.0] = 1.0
        randomMap[randomMap == 0.0] = 0.3
        self.groundTruthMap = randomMap

        # for lidar --> 0 free cell
        #               1 obstacle
        self.ldr = Lidar(r=self.conf["lidar_range"],
                         channels=self.conf["lidar_channels"],
                         map=randomMapOriginal)

        obstacles_idx = np.where(self.groundTruthMap == 1.0)
        obstacles_x = obstacles_idx[0]
        obstacles_y = obstacles_idx[1]
        self.obstacles_idx = np.stack((obstacles_x, obstacles_y), axis=1)
        self.obstacles_idx = [list(i) for i in self.obstacles_idx]

        # 0 if not visible/visited, 1 if visible/visited
        self.exploredMap = np.zeros(self.SIZE, dtype=np.double)

        self.x, self.y = self.conf["start"][0], self.conf["start"][1]

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

        self.viewerActive = False
        self.action = 0

        return self.new_state


    def action_space_sample(self):
        random = np.random.randint(4)
        return random


    def render(self, mode='human'):

        if not self.viewerActive:
            self.viewer = Viewer(self, self.conf["viewer"])
            self.viewerActive = True

        self.viewer.run()
        # XXX: check why flip axes ... @dkoutras
        return np.swapaxes(self.viewer.get_display_as_array(), 0, 1)


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

        self.action = action
        self._applyRLactions(action)
        self._computeReward()
        self._checkDone()
        self._updateTrajectory()

        info = {}
        return self.new_state, self.reward, self.done, info


    def close(self):
        if self.viewerActive:
            self.viewer.quit()
