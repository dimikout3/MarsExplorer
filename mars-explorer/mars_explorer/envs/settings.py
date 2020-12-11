DEFAULT_CONFIG={
    # ======== TOPOLOGY =======
    #  general configuration for the topology of operational area
    "initial":[0,0],
    "size":[42,42],
    #  configuration regarding the movements of uav
    "movementCost":0.2,

    # ======== ENVIROMENT =======
    # configuration regarding the random map generation
    # absolute number of obstacles, randomly placed in env
    "obstacles":10,
    # if rows/colums activated the obstacles will be placed in a semi random
    # spacing
    "number_rows":None,
    "number_columns":None,
    # noise activated only when row/columns activated
    "noise":[0,0],
    # margins expressed in cell if rows/columns not activated
    "margins":[1, 1],
    # obstacle size expressed in cell if rows/columns not activated
    "obstacle_size":[2,2],

    # ======== SENSORS | LIDAR =======
    "lidar_range":6,
    "lidar_channels":32,

    # ======== VIEWER =========
    "viewer":{"width":42*30,
              "height":42*30,
              "title":"Mars-Explorer-V01",
              "drone_img":'/home/dkoutras/GeneralExplorationPolicy/mars-explorer/tests/img/drone.png',
              "obstacle_img":'/home/dkoutras/GeneralExplorationPolicy/mars-explorer/tests/img/block.png',
              "background_img":'/home/dkoutras/GeneralExplorationPolicy/mars-explorer/tests/img/mars.jpg',
              "light_mask":"/home/dkoutras/GeneralExplorationPolicy/mars-explorer/tests/img/light_350_hard.png",
              "night_color":(100, 100, 100),
              "draw_lidar":True,
              "draw_grid":False,
              "draw_traceline":False
             }
}
