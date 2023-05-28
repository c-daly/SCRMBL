import gym
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import numpy as np
from tensorflow.keras.layers import Conv2D, Flatten
from numpy import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = np.empty((capacity,), dtype=object)
        self.capacity = capacity
        self.index = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done):
        self.buffer[self.index] = (state, action, reward, next_state, done)
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = np.random.choice(self.size, size=batch_size, replace=False)
        state, action, reward, next_state, done = zip(*self.buffer[indices])

        return (np.array(state, dtype=np.float32),
                np.array(action, dtype=np.int32),
                np.array(reward, dtype=np.float32),
                np.array(next_state, dtype=np.float32),
                np.array(done, dtype=np.bool))

class DQNAgentv0:
    def __init__(self, state_space, action_space, env, epsilon=0.9, gamma=0.95, lr=1e-3, act_shape=(1,)):
        self.state_shape = state_space
        self.action_space = action_space
        self.n_actions = action_space
        self.epsilon = epsilon
        self.gamma = gamma
        self.replay_buffer = ReplayBuffer(capacity=1500)
        self.env = env
        self.act_shape = act_shape
        if self.env.scenario.model is not None:
            self.model = env.scenario.model
        else:
            self.model = env.create_model(0.0001)
    def create_model(self, lr):

        # Input layer
        input_layer = Input(shape=self.state_shape)

        # Hidden layers
        hidden_layer1 = Dense(64, activation='relu')(input_layer)
        hidden_layer2 = Dense(64, activation='relu')(hidden_layer1)

        # Output layers
        output_layers = [Dense(4, activation='linear')(hidden_layer2) for _ in range(9)]

        # Create model
        model = tf.keras.Model(inputs=input_layer, outputs=output_layers)
        model.compile(optimizer=Adam(lr=lr), loss='mse')
        model.summary()
        return model

    def create_atari_model(self, lr):
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=4, activation='relu', input_shape=self.state_shape))
        model.add(Conv2D(64, (4, 4), strides=2, activation='relu'))
        model.add(Conv2D(64, (3, 3), strides=1, activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.n_actions, activation='linear'))
        model.compile(optimizer=Adam(lr=lr), loss='mse')
        model.summary()
        return model

    def act(self, state, act_shape=(9,)):
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.n_actions, size=self.act_shape)
            return action
        q_values = self.model.predict(state)
        print(f"q value: {np.max(q_values[0])}")
        action = [np.argmax(value[0]) % 4 for value in q_values]
        return action
    def train(self, replay_buffer, batch_size=10):
        #if len(replay_buffer) < batch_size:
        #    return
        if self.replay_buffer.size < self.batch_size:
            return
        try:
            state, action, reward, next_state, done=self.replay_buffer.sample(self.batch_size)
            action = np.array(action)
            reward = np.array(reward)
            done = np.array(done)
            state = np.array(state)
            next_state = np.array(state)
            target = self.model.predict(state)
            target_next = self.model.predict(next_state)

            for idx in range(self.batch_size):
                if idx >= len(target):
                    break
                if done[idx]:
                    target[idx][action[idx]] = reward[idx]
                else:
                    target_reward = (reward[idx] + self.gamma * np.max(target_next[idx])) * .0001
                    if self.act_shape == (1,):
                        target[idx][action[idx]] = target_reward
                    else:
                        target[idx][0][action[idx]] = target_reward
            #target = (reward + self.gamma * np.max(target_next)) * .0001
            self.model.fit(state, target, epochs=1, verbose=0)
        except Exception as e:
            print(f"error: {e}")

    def learn(self, n_games, batch_size):
        self.batch_size = batch_size
        for i in range(n_games):
            state = self.env.reset()
            total_reward = 0

            while True:
                #action = self.act(state[np.newaxis, :])  # Reshape to (1, 84, 84, 4)
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                # print(f"action/reward: {action}/{reward}")
                #next_state = next_state.reshape(next_state.shape[1:])
                self.replay_buffer.push(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                if done:
                    print(f"Episode: {i}, Reward: {total_reward}")
                    break

                self.train(self.replay_buffer, self.batch_size)

            # Epsilon decay
            if self.epsilon > 0.01:
                self.epsilon *= 0.995


