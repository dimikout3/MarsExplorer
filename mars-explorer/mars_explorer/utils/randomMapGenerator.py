import numpy as np
import matplotlib.pyplot as plt

class Generator:

    # def __init__(self, size=[30, 30], obstacles=10, number_rows=None,
    #              number_columns=None, noise=[0, 0], margins=[0.15, 0.15],
    #              obstacle_size=[0.0, 0.0]):
    def __init__(self, config):
        # obstacles --> number of obstacles in the map
        # margins --> left,right,top,bottom margins as percentages that are free

        self.width = config["size"][1]
        self.height = config["size"][0]
        self.number_rows = config["number_rows"]
        self.number_columns = config["number_columns"]
        self.noise = config["noise"]
        self.margins = config["margins"]
        self.obstacle_size = config["obstacle_size"]
        self.obstacles = config["obstacles"]
        self.size = config["size"]

        self.map = np.zeros((self.size[0],self.size[1]))

        self._obstaclesInitialPoistions()
        if self.number_rows!=None and self.number_columns!=None:
            self._noiseObstaclesPositions()
            self._randomObstacleSize()
        else:
            self._randomObstacleSizeCell()
        # apply to map
        self.map[self.hv, self.wv] = 1


    def _randomObstacleSizeCell(self):
        # determine obstacle size (random towards height and width axis)
        for obstacle in range(self.hv.shape[0]):
            ob_width = np.random.randint(self.obstacle_size[0], self.obstacle_size[1]+1)
            ob_height = np.random.randint(self.obstacle_size[0], self.obstacle_size[1]+1)

            h_ind = np.arange(self.hv[obstacle], self.hv[obstacle]+ob_height)
            w_ind = np.arange(self.wv[obstacle], self.wv[obstacle]+ob_width)
            hv_ind, wv_ind = np.meshgrid(h_ind, w_ind)
            self.hv = np.concatenate((self.hv, hv_ind.reshape(-1)))
            self.wv = np.concatenate((self.wv, wv_ind.reshape(-1)))

            # rX = np.clip(rX, 0, map.shape[0] - 1)
            self.hv = np.clip(self.hv, 0, self.size[0]-1)
            self.wv = np.clip(self.wv, 0, self.size[1]-1)

    def _randomObstacleSize(self):
        # determine obstacle size (random towards height and width axis)
        for obstacle in range(self.hv.shape[0]):
            ob_width = np.random.randint(1, self.obstacle_size[1])
            ob_height = np.random.randint(1, self.obstacle_size[0])

            h_ind = np.arange(self.hv[obstacle], self.hv[obstacle]+ob_height)
            w_ind = np.arange(self.wv[obstacle], self.wv[obstacle]+ob_width)
            hv_ind, wv_ind = np.meshgrid(h_ind, w_ind)
            self.hv = np.concatenate((self.hv, hv_ind.reshape(-1)))
            self.wv = np.concatenate((self.wv, wv_ind.reshape(-1)))

            self.hv = np.clip(self.hv, 0, self.size[0]-1)
            self.wv = np.clip(self.wv, 0, self.size[1]-1)


    def _obstaclesInitialPoistions(self):

        if self.number_rows==None and self.number_columns==None:
            # obstacles are placed randomly in the map
            w_obstacles = np.random.randint(self.margins[0], self.width-1-self.margins[0], self.obstacles)
            h_obstacles = np.random.randint(self.margins[1], self.height-1-self.margins[1], self.obstacles)

            self.hv = h_obstacles
            self.wv = w_obstacles

        else:
            # obstacles are placed on rows and colums and then noise is applied
            w_obstacles = np.linspace(self.margins[1],
                                      self.width-1-self.margins[1],
                                      self.number_columns,dtype=np.int)
            h_obstacles = np.linspace(self.margins[0],
                                      self.height-1-self.margins[0],
                                      self.number_rows,dtype=np.int)

            hv, wv = np.meshgrid(h_obstacles, w_obstacles)

            self.hv = np.concatenate(hv[:])
            self.wv = np.concatenate(wv[:])


    def _noiseObstaclesPositions(self):
        # add noise to existing obstacles
        h_top = np.ceil(self.noise[0])
        h_btm = -np.floor(self.noise[0])
        if h_btm >= h_top:h_top+=1
        w_top = np.ceil(self.noise[1])
        w_btm = -np.floor(self.noise[1])
        if w_btm >= w_top:w_top+=1

        self.hv += np.random.randint(h_btm, h_top, size = self.hv.shape[0])
        np.clip(self.hv, 0, self.height-1, out=self.hv)
        self.wv += np.random.randint(w_btm, w_top, size = self.wv.shape[0])
        np.clip(self.wv, 0, self.width-1, out=self.wv)


    def get_map(self):
        return self.map
