from scenarios import BaseScenario
import numpy as np
from gym.spaces import MultiDiscrete, Box, Discrete
from PIL import Image
from utils.ImageUtils import ImageUtils

class MoveToBeaconScenario:
    def __init__(self):
        self.marines = None
        self.map_name = "MoveToBeacon.SC2Map"
        self.action_space = Discrete(3)
        self.map_high = 64
        self.map = np.zeros((self.map_high, self.map_high), dtype=int)
        self.derived_obs = None
        self.observation_space = Box(
            low=0,
            high=256,
            shape=(64, 64),
            dtype=int
        )
        #self.model = create_standard_model(input_shape=self.observation_space.shape, n_outputs=4)

    def get_marines(self):
        self.marines = [unit for unit in self.raw_obs.observation.raw_data.units if unit.alliance == 1]
        return self.marines

    def get_beacon(self):
        units = self.raw_obs.observation.raw_data.units
        return [unit for unit in units if unit.alliance != 1][0]

    def get_derived_obs_from_raw(self, raw):
        self.raw_obs = raw
        marine = self.get_marines()[0]
        self.get_beacon()
        image = raw.observation.render_data.minimap
        temp = ImageUtils.unpack_rgb_image(image)
        self.obs = temp
        return self.obs

