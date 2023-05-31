import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input, Conv2D
from tensorflow.keras.optimizers import Adam, schedules

class DeepQNetwork:
    def __init__(self, obs_space, action_space):
        lr_schedule = schedules.ExponentialDecay(
            initial_learning_rate=.01,
            decay_steps=50,
            decay_rate=0.5)
        self.model = Sequential()
        self.model.add(Input(shape=(1,obs_space)))
        #self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(action_space, activation='linear'))
        #self.model.compile(loss='mse', optimizer=Adam(learning_rate=lr_schedule))
        self.model.compile(loss='mse', optimizer=Adam(learning_rate=0.00001))
