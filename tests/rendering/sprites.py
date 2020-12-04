import pygame as pg
from tests.rendering.settings import *

class Drone(pg.sprite.Sprite):
    def __init__(self, viewer, env):
        self.groups = viewer.all_sprites
        pg.sprite.Sprite.__init__(self, self.groups)
        self.viewer = viewer
        self.image = viewer.drone_img
        # self.image = pg.Surface((TILESIZE, TILESIZE))
        # self.image.fill(PLAYER_IMG)
        self.rect = self.image.get_rect()
        self.env = env

    def update(self):
        self.rect.x = self.env.x * TILESIZE
        self.rect.y = self.env.y * TILESIZE

class Obstacle(pg.sprite.Sprite):
    def __init__(self, viewer, x, y):
        self.groups = viewer.all_sprites, viewer.walls
        pg.sprite.Sprite.__init__(self, self.groups)
        self.viewer = viewer
        self.image = viewer.obstacle_img
        # self.image = pg.Surface((TILESIZE, TILESIZE))
        # self.image.fill(OBSTACLE_IMG)
        self.rect = self.image.get_rect()
        self.x = x
        self.y = y
        self.rect.x = x * TILESIZE
        self.rect.y = y * TILESIZE

class Background(pg.sprite.Sprite):
    def __init__(self, viewer, x, y):
        self.groups = viewer.all_sprites, viewer.walls
        pg.sprite.Sprite.__init__(self, self.groups)
        self.viewer = viewer
        self.image = viewer.obstacle_img
        # self.image = pg.Surface((TILESIZE, TILESIZE))
        # self.image.fill(OBSTACLE_IMG)
        self.rect = self.image.get_rect()
        self.x = x
        self.y = y
        self.rect.x = x * TILESIZE
        self.rect.y = y * TILESIZE
