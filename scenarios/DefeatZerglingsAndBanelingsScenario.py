import numpy as np
from gym.spaces import MultiDiscrete, Box, Discrete
from PIL import Image
from utils.ImageUtils import ImageUtils
from scenarios import BaseScenario


class DefeatZerglingsAndBanelingsScenario:
    def __init__(self):
        self.raw_obs = None
        self.marines = None
        self.map_name = "DefeatZerglingsAndBanelings.SC2Map"
        self.action_space = MultiDiscrete([4, 4, 4, 4, 4, 4, 4, 4, 4])
        self.map_high = 64
        self.map = np.zeros((self.map_high, self.map_high), dtype=int)
        self.derived_obs = None
        self.observation_space = Box(
            low=0,
            high=256,
            shape=(64, 64),
            dtype=int
        )

    def get_marines(self):
        self.marines = [unit for unit in self.raw_obs.observation.raw_data.units if unit.alliance == 1]
        return self.marines

    def get_derived_obs_from_raw(self, raw):
        self.raw_obs = raw
        image = raw.observation.render_data.minimap
        temp = ImageUtils.unpack_grayscale_image(image)
        self.get_marines()
        return temp

