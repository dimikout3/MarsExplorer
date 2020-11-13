import numpy as np
import matplotlib.pyplot as plt

class Generator:

    def __init__(self, size=[30, 30], obstacles=10, number_rows=None,
                 number_columns=None, noise=[0, 0], margins=[0.15, 0.15],
                 obstacle_size=[0.0, 0.0]):

        # obstacles --> number of obstacles in the map
        # margins --> left,right,top,bottom margins as percentages that are free
        self.map = np.zeros((size[0],size[1]))

        width = size[1]
        height = size[0]

        if number_rows==None and number_columns==None:
            # obstacles are placed randomly in the map
            w_obstacles = np.random.randint(width, number_rows*number_columns)
            h_obstacles = np.random.randint(height, number_rows*number_columns)
        else:
            # obstacles are placed on rows and colums and then noise is applied
            w_obstacles = np.linspace(width*margins[1], width-1-width*margins[1],
                                      number_columns,dtype=np.int)
            h_obstacles = np.linspace(height*margins[0], height-1-height*margins[0],
                                      number_rows,dtype=np.int)

        hv, wv = np.meshgrid(h_obstacles, w_obstacles)
        hv = np.concatenate(hv[:])
        wv = np.concatenate(wv[:])

        # add noise to existing obstacles
        h_top, h_btm = np.ceil(height*noise[0]), -np.floor(height*noise[0])
        if h_btm >= h_top:h_top+=1
        w_top, w_btm = np.ceil(width*noise[1]), -np.floor(width*noise[1])
        if w_btm >= w_top:w_top+=1

        hv += np.random.randint(h_btm, h_top, size = hv.shape[0])
        np.clip(hv, 0, height-1, out=hv)
        wv += np.random.randint(w_btm, w_top, size = wv.shape[0])
        np.clip(wv, 0, width-1, out=wv)

        # determine obstacle size (random towards height and width axis)
        for obstacle in range(hv.shape[0]):
            # BUG:  check if width*obstacle_size[1]>1
            ob_width = np.random.randint(1, width*obstacle_size[1])
            ob_height = np.random.randint(1, height*obstacle_size[0])

            hv = np.concatenate((hv, np.repeat(hv[obstacle], ob_height)))
            wv = np.concatenate((wv, np.arange(wv[obstacle], wv[obstacle]+ob_height)))
            wv = np.concatenate((wv, np.repeat(wv[obstacle], ob_width)))
            hv = np.concatenate((hv, np.arange(hv[obstacle], hv[obstacle]+ob_width)))

        # apply to map
        self.map[hv, wv] = 1


    def get_map(self):
        return self.map
