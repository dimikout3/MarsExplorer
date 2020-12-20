# Project setup
# Video link: https://youtu.be/3UxnelT9aCo
import pygame as pg

import sys, os
import numpy as np

from mars_explorer.render.sprites import *

class Viewer():
    def __init__(self,env, conf):
        pg.init()
        self.env = env
        self.conf = conf
        self.screen = pg.display.set_mode((self.conf["width"], self.conf["height"]))
        pg.display.set_caption(self.conf["title"])
        self.clock = pg.time.Clock()
        pg.key.set_repeat(500, 100)

        self.TILESIZE = int(self.conf["width"]/self.env.sizeX)
        self.LIGHT_RADIUS = (self.TILESIZE*2, self.TILESIZE*2)
        self.LIGHT_GREY = (100, 100, 100)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)

        self.load_data()

        # initialize all variables and do all the setup for a new game
        self.all_sprites = pg.sprite.Group()
        self.walls = pg.sprite.Group()

        for x, y in env.obstacles_idx:
            Obstacle(self, x, y)
        self.player = Drone(self, env)

    def load_data(self):
        self.drone_img = pg.image.load(self.conf["drone_img"]).convert_alpha()
        self.drone_img = pg.transform.scale(self.drone_img, (self.TILESIZE, self.TILESIZE))

        self.obstacle_img = pg.image.load(self.conf["obstacle_img"]).convert_alpha()
        self.obstacle_img = pg.transform.scale(self.obstacle_img, (self.TILESIZE, self.TILESIZE))

        self.bck_img = pg.image.load(self.conf["background_img"]).convert_alpha()
        self.bck_img = pg.transform.scale(self.bck_img, (self.conf["height"], self.conf["width"]))

        # lighting effect
        self.fog = pg.Surface((self.conf["width"], self.conf["height"]))
        self.fog.fill(self.conf["night_color"])
        self.light_mask = pg.image.load(self.conf["light_mask"]).convert_alpha()
        # TODO: check radius, image radius is not on edge of the square
        self.light_mask = pg.transform.scale(self.light_mask, self.LIGHT_RADIUS)
        self.light_rect = self.light_mask.get_rect()

    def run(self):
        self.update()
        self.draw()

    def quit(self):
        pg.quit()
        # sys.exit()

    def update(self):
        # update portion of the game loop
        self.all_sprites.update()

    def render_fog_camera(self):
        # dark everywhere except where the lidar sees every time step
        cameraX = (self.env.x+.5)*self.TILESIZE
        cameraY = (self.env.y+.5)*self.TILESIZE

        self.fog.fill(self.conf["night_color"])
        self.light_rect.center = (cameraX, cameraY)
        print(f"light x:{cameraX} y:{cameraY}")

        self.fog.blit(self.light_mask, self.light_rect)
        self.screen.blit(self.fog, (0, 0), special_flags=pg.BLEND_MULT)

    def render_fog_explored(self):
        # dark everywhere(unexplored), but light on every explored cell
        explored_idx = np.where(self.env.outputMap > .0)
        explored_x = explored_idx[0]
        explored_y = explored_idx[1]
        explored_idx = np.stack((explored_x, explored_y), axis=1)
        explored_idx = [list(i) for i in explored_idx]

        self.fog.fill(self.conf["night_color"])

        for x,y in explored_idx:
            cameraX = (x+.5)*self.TILESIZE
            cameraY = (y+.5)*self.TILESIZE

            self.light_rect.center = (cameraX, cameraY)

            self.fog.blit(self.light_mask, self.light_rect)
        self.screen.blit(self.fog, (0, 0), special_flags=pg.BLEND_MULT)

    def draw_lidar_rays(self):
        thetas, ranges = self.env.ldr.thetas, self.env.ldr.ranges
        currentX = self.env.x*self.TILESIZE+0.5*self.TILESIZE
        currentY = self.env.y*self.TILESIZE+0.5*self.TILESIZE
        xObs = (currentX + self.TILESIZE*ranges*np.cos(thetas)).astype(float)
        yObs = (currentY + self.TILESIZE*ranges*np.sin(thetas)).astype(float)
        for x,y in zip(xObs, yObs):
            pg.draw.line(self.screen, self.RED, (currentX, currentY), (x, y))
            pg.draw.circle(self.screen, self.RED, (x, y), self.TILESIZE/8)

    def draw_grid(self):
        for x in range(0, self.conf["width"], self.TILESIZE):
            pg.draw.line(self.screen, self.LIGHT_GREY, (x, 0), (x, self.conf["height"]))
        for y in range(0, self.conf["height"], self.TILESIZE):
            pg.draw.line(self.screen, self.LIGHT_GREY, (0, y), (self.conf["width"], y))

    def draw_traceline(self):

        if self.env.timeStep>2:
            for step in range(len(self.env.drone_trajectory)-1):

                currentX = self.env.drone_trajectory[-1-step][0]*self.TILESIZE+0.5*self.TILESIZE
                currentY = self.env.drone_trajectory[-1-step][1]*self.TILESIZE+0.5*self.TILESIZE
                prevX = self.env.drone_trajectory[-2-step][0]*self.TILESIZE+0.5*self.TILESIZE
                prevY = self.env.drone_trajectory[-2-step][1]*self.TILESIZE+0.5*self.TILESIZE

                pg.draw.line(self.screen, self.BLUE, (currentX, currentY),
                             (prevX, prevY))

    def draw(self):
        self.screen.blit(self.bck_img, (0, 0))
        if self.conf["draw_grid"]:
            self.draw_grid()
        if self.conf["draw_lidar"]:
            self.draw_lidar_rays()
        if self.conf["draw_traceline"]:
            self.draw_traceline()
        self.all_sprites.draw(self.screen)
        # self.render_fog_camera()
        self.render_fog_explored()
        pg.display.flip()

    def get_display_as_array(self):
        return pg.surfarray.array3d(self.screen)
