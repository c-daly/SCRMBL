import gym
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras.optimizers import Adam
import numpy as np
from gym.spaces import MultiDiscrete, Box, Discrete
from PIL import Image


def create_model_with_multidiscrete_output(input_shape, value_max, num_values, lr=0.0001):
    # Input layer
    input_layer = Input(shape=input_shape)

    # Hidden layers
    hidden_layer1 = Dense(64, activation='relu')(input_layer)
    hidden_layer2 = Dense(64, activation='relu')(hidden_layer1)

    # Output layers
    output_layers = [Dense(value_max, activation='linear')(hidden_layer2) for _ in range(num_values)]

    # Create model
    model = tf.keras.Model(inputs=input_layer, outputs=output_layers)
    model.compile(optimizer=Adam(lr=lr), loss='mse')
    model.summary()
    return model

def create_standard_model(input_shape, n_outputs, lr=0.003):
    # Input layer
    #input_layer = Input(shape=input_shape)

    # Hidden layers
    #hidden_layer1 = Dense(64, activation='relu')(input_layer)
    #hidden_layer2 = Dense(64, activation='relu')(hidden_layer1)

    # Output layers
    #output_layers = [Dense(n_outputs, activation='linear')(hidden_layer2)]

    # Create model
    #model = tf.keras.Model(inputs=input_layer, outputs=output_layers)
    #model.compile(optimizer=Adam(lr=lr), loss='mse')
    #model.summary()

    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(4))
    model.compile(optimizer=Adam(lr=lr), loss='mse')
    model.summary()
    return model
class SC2MiniMapScenario(object):
    def __init__(self):
        self.marines = None
        self.map_name = "DefeatZerglingsAndBanelings.SC2Map"
        self.action_space = MultiDiscrete([4, 4, 4, 4, 4, 4, 4, 4, 4])
        self.map_high = 64
        self.map = np.zeros((self.map_high, self.map_high), dtype=int)
        self.derived_obs = None
        self.observation_space = Box(
            low=0,
            high=64,
            shape=(64, 64),
            dtype=int
        )
        #self.model = create_model_with_multidiscrete_output(self.observation_space.shape, 4, 9)
    def unpack_rgb_image(self, plane):
        try:
            """Return a correctly shaped numpy array given the image bytes."""
            #assert plane.bits_per_pixel == 24, "{} != 24".format(plane.bits_per_pixel)
            #size = (64,64)
            #data = np.frombuffer(plane.data, dtype=np.uint8)
            image = Image.frombytes('RGB', (64, 64), plane.data)
            data = np.array(image)
        except Exception as e:
            print(f"unpack error {e}")
        return data

    def unpack_grayscale_image(self, plane):
        """Return a correctly shaped numpy array given the image bytes."""
        # assert plane.bits_per_pixel == 24, "{} != 24".format(plane.bits_per_pixel)
        # size = (64,64)
        # data = np.frombuffer(plane.data, dtype=np.uint8)
        image = Image.frombytes('L', (64, 64), plane.data)
        data = np.array(image)
        return data

    def get_marines(self):
        self.marines = [unit for unit in self.raw_obs.observation.raw_data.units if unit.alliance == 1]
        return self.marines

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
        x_s = []
        y_s = []
        derived_obs = []
        #if len(raw.observation.raw_data.units) > 0:
        #    for unit in raw.observation.raw_data.units:
        #        new_obs = ((unit.pos.x + .001)/8) * ((unit.pos.y + .001)/8)
        #        new_obs = [unit.pos.x, unit.pos.y]
        #        derived_obs.append(new_obs)
        #self.derived_obs = self.print_pixelmap()
        image = raw.observation.render_data.minimap
        temp = self.unpack_rgb_image(image)
        self.get_marines()
        return temp
        #return self.derived_obs

class MoveToBeaconScenario(object):
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
        self.model = create_standard_model(input_shape=self.observation_space.shape, n_outputs=4)

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

class DefeatRoachesScenario(object):
    def __init__(self):
        self.marines = None
        self.map_name = "DefeatRoaches.SC2Map"
        self.action_space = MultiDiscrete([4, 4, 4, 4, 4, 4, 4, 4, 4])
        self.map_high = 64
        self.map = np.zeros((self.map_high, self.map_high), dtype=int)
        self.derived_obs = None
        self.observation_space = Box(
            low=0,
            high=64,
            shape=(2, 2),
            dtype=np.int
        )

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
        x_s = []
        y_s = []
        derived_obs = []
        #if len(raw.observation.raw_data.units) > 0:
        #    for unit in raw.observation.raw_data.units:
        #        new_obs = ((unit.pos.x + .001)/8) * ((unit.pos.y + .001)/8)
        #        new_obs = [unit.pos.x, unit.pos.y]
        #        derived_obs.append(new_obs)
        self.derived_obs = self.print_pixelmap()
        return self.derived_obs
