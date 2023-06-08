from scenarios import BaseScenario
from utils.ImageUtils import ImageUtils
import numpy as np
from gym.spaces import MultiDiscrete, Box, Discrete
class DefeatRoachesScenario:
    def __init__(self):
        self.marines = None
        self.map_name = "DefeatRoaches.SC2Map"
        self.action_space = MultiDiscrete([9, 9, 9, 9, 9, 9, 9, 9, 9])
        self.map_high = 64
        self.map = np.zeros((self.map_high, self.map_high), dtype=int)
        self.enemies = None
        self.derived_obs = None
        self.raw_obs = None
        self.observation_space = Box(
            low=0,
            high=64,
            shape=(2, 2),
            dtype=int
        )


    def get_marines(self):
        self.marines = [unit for unit in self.raw_obs.observation.raw_data.units if unit.alliance == 1]
        return self.marines

    def get_enemies(self):
        self.enemies = [unit for unit in self.raw_obs.observation.raw_data.units if unit.alliance != 1]
        return self.enemies

    def get_derived_obs_from_raw(self, raw):
        self.raw_obs = raw
        image = raw.observation.render_data.minimap
        temp = ImageUtils.unpack_rgb_image(image)
        self.get_marines()
        self.get_enemies()
        return temp