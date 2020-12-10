from mars_explorer.envs.settings.settings import DEFAULT_CONFIG as conf
from mars_explorer.utils.randomMapGenerator import Generator
import matplotlib.pyplot as plt

g=Generator(conf)
plt.imshow(g.get_map())
plt.show()
