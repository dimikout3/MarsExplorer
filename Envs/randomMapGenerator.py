import numpy as np
import matplotlib.pyplot as plt

class Generator:

    def __init__(self, size=[30, 30], obstacles=10, number_rows=None,
                 number_columns=None, noise=[0, 0], margins=[0.15, 0.15],
                 obstacle_size=[0.0, 0.0]):

        # obstacles --> number of obstacles in the map
        # margins --> left,right,top,bottom margins as percentages that are free
        self.map = np.zeros((size[0],size[1]))

        self.width = size[1]
        self.height = size[0]
        self.number_rows = number_rows
        self.number_columns = number_columns
        self.noise = noise
        self.margins = margins
        self.obstacle_size = obstacle_size
        self.obstacles = obstacles
        self.size = size

        self._obstaclesInitialPoistions()
        self._noiseObstaclesPositions()
        self._randomObstacleSize()

        # apply to map
        self.map[self.hv, self.wv] = 1


    def _randomObstacleSize(self):
        # determine obstacle size (random towards height and width axis)
        for obstacle in range(self.hv.shape[0]):
            # BUG:  check if width*obstacle_size[1]>1
            ob_width = np.random.randint(1, self.width*self.obstacle_size[1])
            ob_height = np.random.randint(1, self.height*self.obstacle_size[0])

            self.hv = np.concatenate((self.hv, np.repeat(self.hv[obstacle], ob_height)))
            self.wv = np.concatenate((self.wv, np.arange(self.wv[obstacle], self.wv[obstacle]+ob_height)))
            self.wv = np.concatenate((self.wv, np.repeat(self.wv[obstacle], ob_width)))
            self.hv = np.concatenate((self.hv, np.arange(self.hv[obstacle], self.hv[obstacle]+ob_width)))


    def _obstaclesInitialPoistions(self):

        if self.number_rows==None and self.number_columns==None:
            # obstacles are placed randomly in the map
            w_obstacles = np.random.randint(self.width,
                                            self.number_rows*self.number_columns)
            h_obstacles = np.random.randint(self,
                                            self.number_rows*self.number_columns)
        else:
            # obstacles are placed on rows and colums and then noise is applied
            w_obstacles = np.linspace(self.width*self.margins[1],
                                      self.width-1-self.width*self.margins[1],
                                      self.number_columns,dtype=np.int)
            h_obstacles = np.linspace(self.height*self.margins[0],
                                      self.height-1-self.height*self.margins[0],
                                      self.number_rows,dtype=np.int)

        hv, wv = np.meshgrid(h_obstacles, w_obstacles)
        self.hv = np.concatenate(hv[:])
        self.wv = np.concatenate(wv[:])


    def _noiseObstaclesPositions(self):
        # add noise to existing obstacles
        h_top = np.ceil(self.height*self.noise[0])
        h_btm = -np.floor(self.height*self.noise[0])
        if h_btm >= h_top:h_top+=1
        w_top = np.ceil(self.width*self.noise[1])
        w_btm = -np.floor(self.width*self.noise[1])
        if w_btm >= w_top:w_top+=1

        self.hv += np.random.randint(h_btm, h_top, size = self.hv.shape[0])
        np.clip(self.hv, 0, self.height-1, out=self.hv)
        self.wv += np.random.randint(w_btm, w_top, size = self.wv.shape[0])
        np.clip(self.wv, 0, self.width-1, out=self.wv)


    def get_map(self):
        return self.map