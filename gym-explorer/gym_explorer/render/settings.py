# define some colors (R, G, B)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DARKGREY = (40, 40, 40)
LIGHTGREY = (100, 100, 100)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)

# game settings
WIDTH = 42*30   # 16 * 64 or 32 * 32 or 64 * 16
HEIGHT = 42*30  # 16 * 48 or 32 * 24 or 64 * 12
FPS = 60
TITLE = "Star-CAS-V01"

DRONE_IMG = 'img/drone.png'
OBSTACLE_IMG = 'img/block.png'
BG_IMG = 'img/mars.jpg'
BGCOLOR = DARKGREY

GRIDWIDTH = 42
GRIDHEIGHT = 42
TILESIZE = int(WIDTH/GRIDWIDTH)

NIGHT_COLOR = (10, 10, 10)
# 6 --> lidar radius
# TODO: check radius, image radius is not on edge of the square
LIGHT_RADIUS = (TILESIZE*2, TILESIZE*2)
LIGHT_MASK = "img/light_350_hard.png"
