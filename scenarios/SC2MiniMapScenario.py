import gym
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras.optimizers import Adam
import numpy as np
from gym.spaces import MultiDiscrete, Box, Discrete
from PIL import Image
from utils.ImageUtils import ImageUtils

#
# def create_model_with_multidiscrete_output(input_shape, value_max, num_values, lr=0.0001):
#     # Input layer
#     input_layer = Input(shape=input_shape)
#
#     # Hidden layers
#     hidden_layer1 = Dense(64, activation='relu')(input_layer)
#     hidden_layer2 = Dense(64, activation='relu')(hidden_layer1)
#
#     # Output layers
#     output_layers = [Dense(value_max, activation='linear')(hidden_layer2) for _ in range(num_values)]
#
#     # Create model
#     model = tf.keras.Model(inputs=input_layer, outputs=output_layers)
#     model.compile(optimizer=Adam(lr=lr), loss='mse')
#     model.summary()
#     return model
#
# def create_standard_model(input_shape, n_outputs, lr=0.003):
#     # Input layer
#     #input_layer = Input(shape=input_shape)
#
#     # Hidden layers
#     #hidden_layer1 = Dense(64, activation='relu')(input_layer)
#     #hidden_layer2 = Dense(64, activation='relu')(hidden_layer1)
#
#     # Output layers
#     #output_layers = [Dense(n_outputs, activation='linear')(hidden_layer2)]
#
#     # Create model
#     #model = tf.keras.Model(inputs=input_layer, outputs=output_layers)
#     #model.compile(optimizer=Adam(lr=lr), loss='mse')
#     #model.summary()
#
#     model = Sequential()
#     model.add(Input(shape=input_shape))
#     model.add(Dense(32, activation='relu'))
#     model.add(Dense(4))
#     model.compile(optimizer=Adam(lr=lr), loss='mse')
#     model.summary()
#     return model
class SC2MiniMapScenario(object):
    def __init__(self):
        self.raw_obs = None
        self.marines = None
        self.map_name = "DefeatZerglingsAndBanelings.SC2Map"
        #self.action_space = MultiDiscrete([4, 4, 4, 4, 4, 4, 4, 4, 4])
        self.action_space = MultiDiscrete([5, 5, 5, 5, 5, 5, 5, 5, 5])
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

    #def print_pixelmap(self):
    #    self.get_marines()
    #    self.map = np.zeros((self.map_high, self.map_high), dtype=int)
    #    for i, marine in enumerate(self.marines):
    #        x = int(marine.pos.x)
    #        y = int(marine.pos.y)
    #        self.map[x][y] = i + 1

    #    beacon = self.get_beacon()
    #    self.map[int(beacon.pos.x)][int(beacon.pos.y)] = -1
    #    x = 5
    #    return self.map

    def get_derived_obs_from_raw(self, raw):
        self.raw_obs = raw
        image = raw.observation.render_data.minimap
        temp = ImageUtils.unpack_grayscale_image(image)
        self.get_marines()
        return temp

