import pygame as pg
import numpy as np

class Drone(pg.sprite.Sprite):
    def __init__(self, viewer, env):
        self.groups = viewer.all_sprites
        pg.sprite.Sprite.__init__(self, self.groups)
        self.env = env
        self.viewer = viewer
        self.rotate_img()
        self.rect = self.image.get_rect()

    def update(self):
        self.rotate_img()
        self.rect.x = self.env.x * self.viewer.TILESIZE
        self.rect.y = self.env.y * self.viewer.TILESIZE

    def rotate_img(self):
        if self.env.action == 0:
            self.image = pg.transform.rotate(self.viewer.drone_img, 90)
        elif self.env.action == 1:
            self.image = pg.transform.rotate(self.viewer.drone_img, -90)
        elif self.env.action == 2:
            self.image = pg.transform.rotate(self.viewer.drone_img, 0)
        elif self.env.action == 3:
            self.image = pg.transform.rotate(self.viewer.drone_img, 180)

class Obstacle(pg.sprite.Sprite):
    def __init__(self, viewer, x, y):
        self.groups = viewer.all_sprites
        pg.sprite.Sprite.__init__(self, self.groups)
        self.viewer = viewer
        self.image = viewer.obstacle_img
        # self.image = pg.Surface((TILESIZE, TILESIZE))
        # self.image.fill(OBSTACLE_IMG)
        self.rect = self.image.get_rect()
        self.x = x
        self.y = y
        self.rect.x = x * self.viewer.TILESIZE
        self.rect.y = y * self.viewer.TILESIZE
