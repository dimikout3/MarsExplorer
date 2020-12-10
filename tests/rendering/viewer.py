# Project setup
# Video link: https://youtu.be/3UxnelT9aCo
import pygame as pg

import sys, os
import numpy as np
# sys.path.append(os.path.join(os.getcwd(), "tests/rendering"))

from tests.rendering.settings import *
from tests.rendering.sprites import *

class Viewer():
    def __init__(self,env):
        pg.init()
        self.env = env
        self.screen = pg.display.set_mode((WIDTH, HEIGHT))
        pg.display.set_caption(TITLE)
        self.clock = pg.time.Clock()
        pg.key.set_repeat(500, 100)
        self.load_data()

        # initialize all variables and do all the setup for a new game
        self.all_sprites = pg.sprite.Group()
        self.walls = pg.sprite.Group()

        # for x in range(0, env.sizeX):
        #     for y in range(0, env.sizeY):
        #         Background(self, x, y)
        for x, y in env.obstacles_idx:
            Obstacle(self, x, y)
        self.player = Drone(self, env)

    def load_data(self):
        img_folder = os.path.join(os.getcwd(), '../render/images')

        self.drone_img = pg.image.load(os.path.join(img_folder, DRONE_IMG)).convert_alpha()
        self.drone_img = pg.transform.scale(self.drone_img, (TILESIZE, TILESIZE))

        self.obstacle_img = pg.image.load(os.path.join(img_folder, OBSTACLE_IMG)).convert_alpha()
        self.obstacle_img = pg.transform.scale(self.obstacle_img, (TILESIZE, TILESIZE))

        self.background_img = pg.image.load(os.path.join(img_folder, BACKGROUND_IMG)).convert_alpha()
        self.background_img = pg.transform.scale(self.background_img, (TILESIZE, TILESIZE))

        self.bck_img = pg.image.load(os.path.join(img_folder, BG_IMG)).convert_alpha()
        self.bck_img = pg.transform.scale(self.bck_img, (HEIGHT, WIDTH))

    def run(self):
        self.update()
        self.draw()

    def quit(self):
        pg.quit()
        sys.exit()

    def update(self):
        # update portion of the game loop
        self.all_sprites.update()

    def draw_lidar_rays(self):
        thetas, ranges = self.env.ldr.thetas, self.env.ldr.ranges
        currentX = self.env.x*TILESIZE+0.5*TILESIZE
        currentY = self.env.y*TILESIZE+0.5*TILESIZE
        xObs = (currentX + TILESIZE*ranges*np.cos(thetas)).astype(float)
        yObs = (currentY + TILESIZE*ranges*np.sin(thetas)).astype(float)
        for x,y in zip(xObs, yObs):
            pg.draw.line(self.screen, RED, (currentX, currentY), (x, y))
            pg.draw.circle(self.screen, RED, (x, y), TILESIZE/8)

    def draw_grid(self):
        for x in range(0, WIDTH, TILESIZE):
            pg.draw.line(self.screen, LIGHTGREY, (x, 0), (x, HEIGHT))
        for y in range(0, HEIGHT, TILESIZE):
            pg.draw.line(self.screen, LIGHTGREY, (0, y), (WIDTH, y))

    def draw(self):
        self.screen.blit(self.bck_img, (0, 0))
        # self.draw_grid()
        self.draw_lidar_rays()
        self.all_sprites.draw(self.screen)
        pg.display.flip()

    def show_start_screen(self):
        pass

    def show_go_screen(self):
        pass
