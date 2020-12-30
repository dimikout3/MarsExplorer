DEFAULT_CONFIG={
    # ======== TOPOLOGY =======
    #  general configuration for the topology of operational area
    "initial":[0,0],
    "size":[21,21],
    #  configuration regarding the movements of uav
    "movementCost":0.2,

    # ======== ENVIROMENT =======
    # configuration regarding the random map generation
    # absolute number of obstacles, randomly placed in env
    "obstacles":12,
    # if rows/colums activated the obstacles will be placed in a semi random
    # spacing
    "number_rows":None,
    "number_columns":None,
    # noise activated only when row/columns activated
    # maximum noise on each axes
    "noise":[0,0],
    # margins expressed in cell if rows/columns not activated
    "margins":[1, 1],
    # obstacle size expressed in cell if rows/columns not activated
    "obstacle_size":[2,2],
    # mas number of steps for the environment
    "max_steps":400,
    "bonus_reward":400,
    "collision_reward":-400,
    "out_of_bounds_reward":-400,

    # ======== SENSORS | LIDAR =======
    "lidar_range":6,
    "lidar_channels":32,

    # ======== VIEWER =========
    "viewer":{"width":21*30,
              "height":21*30,
              "title":"Mars-Explorer-V01",
              "drone_img":'/home/dkoutras/GeneralExplorationPolicy/mars-explorer/tests/img/drone.png',
              "obstacle_img":'/home/dkoutras/GeneralExplorationPolicy/mars-explorer/tests/img/block.png',
              "background_img":'/home/dkoutras/GeneralExplorationPolicy/mars-explorer/tests/img/mars.jpg',
              "light_mask":"/home/dkoutras/GeneralExplorationPolicy/mars-explorer/tests/img/light_350_hard.png",
              "night_color":(20, 20, 20),
              "draw_lidar":True,
              "draw_grid":False,
              "draw_traceline":False
             }
}
