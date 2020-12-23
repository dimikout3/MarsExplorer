import numpy as np
import matplotlib.pyplot as plt

class Grid:

    def __init__(self, size=[10,10], movementCost=0.2, renderingThreshold=np.inf):

        self.sizeX = size[0]
        self.sizeY = size[1]

        self.movementCost = movementCost

        self.SIZE = size

        self.renderingThreshold = renderingThreshold


    def reset(self, start=[0,0]):

        self.maxSteps = self.sizeX * self.sizeY * 1.5

        # groundTruthMap --> 0 where free to move, 1 obstacle
        self.groundTruthMap = np.zeros(self.SIZE, dtype=np.double)

        # 0 if not visible/visited, 1 if visible/visited
        self.exploredMap = np.zeros(self.SIZE, dtype=np.double)

        self.x, self.y = start[0], start[1]

        self.maxEpisodeReward = np.count_nonzero(self.groundTruthMap)*2 - \
                                np.count_nonzero(self.groundTruthMap)*self.movementCost

        self.state_trajectory = []
        self.reward_trajectory = []

        # starting position is explored
        self.exploredMap[self.x, self.y] = 1
        self.outputMap = self.exploredMap.copy()
        self.outputMap[self.x, self.y] = 0.5
        self.new_state = [self.outputMap]
        self.reward = 0
        self.done = False
        self.timeStep = 0

        return self.new_state


    def action_space_sample(self):
        random = np.random.randint(4)
        return random


    def _checkRendering(self):

        renderingFromThreshold = np.sum(self.reward_trajectory) >= self.renderingThreshold
        if self.done and renderingFromThreshold:
            self.render()


    def render(self, path = 'experiments'):

        for step,image in enumerate(self.state_trajectory):
            plt.imshow(image[0])
            plt.title(f"Time Step: {step}")
            plt.ylabel("Y-Axis")
            plt.xlabel("X-Axis")
            plt.savefig(f"{path}/step_{step}.png")
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

        self.x += x
        self.y += y

        # If we are out of bounds, fix!
        if self.x < 0:
            self.x = 0
        elif self.x > self.sizeX-1:
            self.x = self.sizeX-1
        if self.y < 0:
            self.y = 0
        elif self.y > self.sizeY-1:
            self.y = self.sizeY-1


    def _updateMaps(self):

        self.pastExploredMap = self.exploredMap.copy()
        self.exploredMap[self.x, self.y] = 1


    def _applyRLactions(self,action):

        self._choice(action)
        self._updateMaps()

        self.outputMap = self.exploredMap.copy()
        self.outputMap[self.x, self.y] = 0.5
        self.new_state = [self.outputMap]
        self.timeStep += 1


    def _computeReward(self):

        pastExploredCells = np.count_nonzero(self.pastExploredMap)
        currentExploredCells = np.count_nonzero(self.exploredMap)

        # TODO: add fixed cost for moving (-0.5 per move)
        self.reward = currentExploredCells - pastExploredCells - self.movementCost


    def _checkDone(self):

        if self.timeStep > self.maxSteps:
            self.done = True
        elif np.count_nonzero(self.exploredMap) == self.SIZE[0]**2:
            self.done = True
            self.reward = self.sizeX * self.sizeY
        else:
            self.done = False

        # if self.done:
        #     # added bonus for full exploration
        #     maxReward = self.sizeX * self.sizeY
        #
        #     exploredCells = np.count_nonzero(self.exploredMap)
        #     allCells = self.sizeX * self.sizeY
        #     percentageExplored = exploredCells/allCells
        #
        #     self.reward += maxReward*percentageExplored


    def _updateTrajectory(self):

        self.state_trajectory.append(self.new_state)
        self.reward_trajectory.append(self.reward)


    def step(self, action):

        self._applyRLactions(action)
        self._computeReward()
        self._checkDone()
        self._updateTrajectory()
        self._checkRendering()

        return self.new_state, self.reward, self.done
