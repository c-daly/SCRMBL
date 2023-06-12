import numpy as np
from gym.spaces import MultiDiscrete, Box, Discrete
from PIL import Image
from utils.ImageUtils import ImageUtils
from scenarios import BaseScenario
from spaces import SpaceContainer


class DefeatZerglingsAndBanelingsScenario:
    def __init__(self):
        self.raw_obs = None
        self.marines = None
        self.enemies = None
        self.map_name = "DefeatZerglingsAndBanelings.SC2Map"
        self.action_space = MultiDiscrete([15, 15, 15, 15, 15, 15, 15, 15, 15])
        self.map_high = 64
        self.map = np.zeros((self.map_high, self.map_high), dtype=int)
        self.derived_obs = None
        self.observation_space = Box(
            low=0,
            high=256,
            shape=(64, 64, 3),
            dtype=np.uint8
        )
        self.space_container = SpaceContainer(self.observation_space, self.action_space)

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

