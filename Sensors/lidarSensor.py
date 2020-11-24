import numpy as np
import matplotlib.pyplot as plt

class Lidar:

    def __init__(self, r=np.inf, fov=[0,2*np.pi], channels=8):
        # channels --> number of arrays per scan
        self.maxRange = r
        self.fov = fov
        self.channels = channels

        self.thetas = np.linspace(0, 2*np.pi, channels, endpoint=False)
        self.threshold = np.sqrt(2*0.5**2)


    def scan(self, map, position, resolution=200, dtype=np.int):
        # position --> (height, width) of sensor position (robots)
        # map --> ground truth, with FULL visibility
        p1 = np.array(position)

        p2 = np.stack((p1[0]+self.maxRange*np.cos(self.thetas),
                       p1[1]+self.maxRange*np.sin(self.thetas)),axis=1)

        # rX = np.linspace(p1[0], p2[:,0], resolution, dtype=dtype)
        # rY = np.linspace(p1[1], p2[:,1], resolution, dtype=dtype)
        rX = np.linspace(p1[0], p2[:,0], resolution)
        rY = np.linspace(p1[1], p2[:,1], resolution)
        rX = np.round(rX).astype(dtype)
        rY = np.round(rY).astype(dtype)

        rays = np.stack((rX, rY), axis=2)
        # XXX: consider using np.unique(rays, axis=0?) to avoid repetition

        ranges = np.repeat(self.maxRange, self.channels)

        for channel in range(self.channels):

            rays_channel = np.clip(rays[:,channel,:], 0, np.min(map.shape))

            # where the channel's ray is in obstacle
            obstacles_idx = np.where(map[rays_channel[:,0], rays_channel[:,1]])[0]
            # import pdb; pdb.set_trace()
            if obstacles_idx.size>0:
                p1_repeated = np.tile(p1, (obstacles_idx.size,1))
                distances = np.linalg.norm( p1_repeated - rays_channel[obstacles_idx], axis=1)
                ranges[channel] = np.min(distances)

        return self.thetas, ranges
