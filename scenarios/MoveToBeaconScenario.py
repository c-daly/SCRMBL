import BaseScenario
class MoveToBeaconScenario(BaseScenario):
    def __init__(self):
        self.marines = None
        self.map_name = "MoveToBeacon.SC2Map"
        self.action_space = Discrete(4)
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

    def unpack_rgb_image(self, plane):
        """Return a correctly shaped numpy array given the image bytes."""
        #assert plane.bits_per_pixel == 24, "{} != 24".format(plane.bits_per_pixel)
        #size = (64,64)
        #data = np.frombuffer(plane.data, dtype=np.uint8)
        image = Image.frombytes('L', (64, 64), plane.data)
        data = np.array(image)
        return data

    def get_marines(self):
        self.marines = [unit for unit in self.raw_obs.observation.raw_data.units if unit.alliance == 1]
        return self.marines

    def get_beacon(self):
        units = self.raw_obs.observation.raw_data.units
        return [unit for unit in units if unit.alliance != 1][0]

    def print_pixelmap(self):
        self.get_marines()
        self.map = np.zeros((self.map_high, self.map_high), dtype=int)
        for i, marine in enumerate(self.marines):
            x = int(marine.pos.x)
            y = int(marine.pos.y)
            self.map[x][y] = i + 1

        beacon = self.get_beacon()
        self.map[int(beacon.pos.x)][int(beacon.pos.y)] = -1
        x = 5
        return self.map

    def get_derived_obs_from_raw(self, raw):
        self.raw_obs = raw
        marine = self.get_marines()[0]
        #beacon = self.get_beacon()
        x_s = []
        y_s = []
        derived_obs = []
        #if len(raw.observation.raw_data.units) > 0:
        #    for unit in raw.observation.raw_data.units:
        #        new_obs = ((unit.pos.x + .001)/8) * ((unit.pos.y + .001)/8)
        #        new_obs = [unit.pos.x, unit.pos.y]
        #        derived_obs.append(new_obs)
        #self.obs = (([int(marine.pos.x), int(marine.pos.y)], [int(beacon.pos.x), int(beacon.pos.y)]))
        #print(self.obs)
        image = raw.observation.render_data.minimap
        temp = self.unpack_rgb_image(image)
        self.obs = temp
        return self.obs

