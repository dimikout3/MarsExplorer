import numpy as np

class Grid:

    def __init__(self, size=[10,10], start=[0,0], maxSteps=100):

        self.sizeX = size[0]
        self.sizeY = size[1]

        self.SIZE = size

        self.maxSteps = maxSteps

        # groundTruthMap --> 0 where free to move, 1 obstacle
        self.groundTruthMap = np.zeros(size)

        # 0 if not visible/visited, 1 if visible/visited
        self.exploredMap = np.zeros(size)

        self.x, self.y = start[0], start[1]

        # starting position is explored
        self.exploredMap[self.x, self.y] = 1

        self.new_state = (self.x, self.y)
        self.reward = 0
        self.done = False
        self.timeStep = 0


    def reset(self, start=[0,0], maxSteps=100):

        self.maxSteps = maxSteps
        
        self.x, self.y = start[0], start[1]

        self.new_state = (self.x, self.y)
        self.reward = 0
        self.done = False
        self.timeStep = 0

        self.exploredMap = np.zeros(size)
        self.exploredMap[self.x, self.y] = 1

        return self.new_state


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

        self.new_state = (self.x, self.y)
        self.timeStep += 1


    def _computeReward(self):

        pastExploredCells = np.count_nonzero(self.pastExploredMap)
        currentExploredCells = np.count_nonzero(self.exploredMap)

        # TODO: add fixed cost for moving (-0.5 per move)
        self.reward = currentExploredCells - pastExploredCells - 0.2


    def _checkDone(self):

        if self.timeStep > self.maxSteps:
            self.done = True

        elif np.count_nonzero(self.exploredMap) == self.SIZE[0]**2:
            self.done = True
            # added bonus for full exploration
            self.reward += 100

        else:
            self.done = False


    def step(self, action):

        self._applyRLactions(action)

        self._computeReward()

        self._checkDone()

        return self.new_state, self.reward, self.done
