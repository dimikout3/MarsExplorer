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

        hv, wv = np.where(map>0.5)
        p3 =np.stack((hv,wv),axis=1)

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


    def scan_line_circle(self, map, position):
        # position --> (height, width) of sensor position (robots)
        # map --> ground truth, with FULL visibility

        def line_circle_intercect(p1, p2, q, r):
            # https://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm
            u = p2 - p1

            a = np.dot(u,u)
            b = 2*(np.dot(u, p1-q))
            c = np.dot(p1,p1) + np.dot(q,q) - 2*np.dot(p1,q) - r**2

            v1 = q-p1
            v2 = p2-p1
            phi = np.arccos(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))
            disc = b**2 - 4 * a * c
            if disc<.0 and phi<np.pi:
                return False
            else:
                return True

        p1 = np.array(position)

        p2 = np.stack((p1[0]+self.maxRange*np.cos(self.thetas),
                       p1[1]+self.maxRange*np.sin(self.thetas)),axis=1)

        hv, wv = np.where(map>0.5)
        p3 =np.stack((hv,wv),axis=1)

        p1_repeated = np.repeat(p1, p3.shape[0])

        r = np.repeat(self.maxRange, self.channels)

        for channel in range(self.channels):
            p2_channel = p2[channel]

            p2_channel_repeated = np.repeat(p2_channel, p3.shape[0])

            # intercect = np.array(list(map(line_circle_intercect, p1_repeated,
            #                               p2_channel_repeated, p3, radius)))
            intercect = []
            for i in range(p3.shape[0]):
                intercect.append(line_circle_intercect(p1, p2_channel,
                                                       p3[i], self.threshold))

            obstacles_idx = np.where(intercect)[0]
            if obstacles_idx.size > 0:
                r_channel = np.min(np.linalg.norm(p3[obstacles_idx]-p1, axis=1))
                r_channel = np.min((r_channel, self.maxRange))

                r[channel] = r_channel

        # import pdb; pdb.set_trace()
        return self.thetas, r


    def scan_legacy(self, map, position):
        # position --> (height, width) of sensor position (robots)
        # map --> ground truth, with FULL visibility
        p1 = np.array(position)

        p2 = np.stack((p1[0]+self.maxRange*np.cos(self.thetas),
                       p1[1]+self.maxRange*np.sin(self.thetas)),axis=1)

        hv, wv = np.where(map>0.5)
        p3 =np.stack((hv,wv),axis=1)

        r = []

        for channel in range(self.channels):
            p2_channel = p2[channel]

            d = np.abs(np.cross(p2_channel-p1, p1-p3))/np.linalg.norm(p2_channel-p1)

            obstacles_idx = np.where(d<self.threshold)[0]
            if obstacles_idx.size > 0:
                r_channel = np.min(np.linalg.norm(p3[obstacles_idx]-p1, axis=1))
                r_channel = np.min((r_channel, self.maxRange))
            else:
                r_channel = self.maxRange
            r.append(r_channel)

        r = np.array(r)
        # import pdb; pdb.set_trace()
        return self.thetas, r
