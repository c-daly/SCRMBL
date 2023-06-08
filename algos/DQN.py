import numpy as np
import random
from networks.DeepQNetwork import DeepQNetwork
from networks.CNN import CNN
from collections import deque
import gym
import tensorflow as tf
from keras.utils import np_utils

from tensorflow.keras.utils import to_categorical
# Define the Q-Network
#class DQNetwork:
#    def __init__(self, obs_space, action_space):
#        lr_schedule = schedules.ExponentialDecay(
#            initial_learning_rate=.01,
#            decay_steps=50,
#            decay_rate=0.5)
#        self.model = Sequential()
#        self.model.add(Input(shape=(1,obs_space)))
#        #self.model.add(Dense(64, activation='relu'))
#        self.model.add(Dense(64, activation='relu'))
#        self.model.add(Dense(action_space, activation='linear'))
#        #self.model.compile(loss='mse', optimizer=Adam(learning_rate=lr_schedule))
#        self.model.compile(loss='mse', optimizer=Adam(learning_rate=0.00001))

#class ConvDQNetwork:
#    def __init__(self, obs_space, action_space):
#        # Network defined by the Deepmind paper
#        lr_schedule = schedules.ExponentialDecay(
#            initial_learning_rate=.01,
#            decay_steps=50,
#            decay_rate=0.5)
#
#        self.model = Sequential()
#        self.model.add(Input(shape=(1, 64, 64, 3)))
#        #self.model.add(Input(shape=(1,obs_space)))
#
#        # Convolutions on the frames on the screen
#        self.model.add(Conv2D(32, 8, strides=4, activation="relu"))
#        self.model.add(Conv2D(64, 4, strides=2, activation="relu"))
#        self.model.add(Conv2D(64, 3, strides=1, activation="relu"))

#        self.model.add(Flatten())

#        self.model.add(Dense(512, activation="relu"))
#        self.model.add(Dense(36, activation="linear"))
#        self.model.compile(loss='mse', optimizer=Adam(learning_rate=lr_schedule))

# Define Replay Memory
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Define DQN Agent
class DQNAgent:
    def __init__(self, env, obs_space, action_space, batch_size, capacity):
        self.env = env
        self.obs_space = obs_space
        self.obs_space_flat_dim = np.prod(obs_space.shape)
        self.action_space = action_space

        if isinstance(env.action_space, gym.spaces.MultiDiscrete):
            self.action_space_flat_dim = np.prod(action_space.shape) * action_space.nvec[0]
        else:
            self.action_space_flat_dim = env.action_space.n

        self.memory = ReplayMemory(capacity)
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.9995
        self.batch_size = batch_size
        self.capacity = capacity

        #self.network = DeepQNetwork(self.obs_space_flat_dim, self.action_space_flat_dim)
        self.network = CNN(self.obs_space, self.action_space_flat_dim)


    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def act(self, state):

        if isinstance(self.env.action_space, gym.spaces.MultiDiscrete):
            #size = (self.env.action_space.nvec[0], self.action_space_flat_dim)
            size = (1, self.action_space.shape[0], self.action_space.nvec[0])
        else:
            size = (1, self.action_space_flat_dim)
            random_choice = np.random.choice(4, size=(1,))

        if np.random.rand() <= self.epsilon:
            if isinstance(self.env.action_space, gym.spaces.Discrete):
                return random_choice
            return np.random.choice(self.action_space.nvec[0], size=(len(self.action_space.nvec)))
        reshaped_state = np.expand_dims(state, axis=0)
        net_result = self.network.model.predict(reshaped_state, verbose=0)
        #net_result = self.network.model.predict(state, verbose=0)
        #action = np.argmax(net_result.reshape(size, axis=1))
        action = net_result.reshape(size)
        #action = np.amax(action[0], axis=1)

        if isinstance(self.env.action_space, gym.spaces.MultiDiscrete):
            action = np.argmax(action[0], axis=1)
        else:
            action = np.argmax(action)
        return action

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        try:
            result = self.memory.sample(self.batch_size)
            state = np.array([a[0] for a in result])
            state = state.reshape(self.batch_size, 64, 64, 3)
            #state = state.reshape(self.batch_size, 1, self.obs_space_flat_dim)
            next_state = np.array([a[3] for a in result])
            #next_state = next_state.reshape(self.batch_size, 1, self.obs_space_flat_dim)
            next_state = next_state.reshape(self.batch_size, 64, 64, 3)
            done = [int(a[4]) for a in result]
            #done_ints = np.zeros(np.shape(done))
            #dones = to_categorical(done)
            reward = [a[2] for a in result]
            target = self.network.model.predict(state, verbose=0)
            #target = target.reshape(self.batch_size, 15, 15)
            #test_target = to_categorical(target)
            target_next = self.network.model.predict(next_state, verbose=0)
            #target_next = target_next.reshape(self.batch_size, 15, 15)
            #target_next = np.reshape(target_next, (self.batch_size, 225))
            #target += self.gamma * target_next * done_ints.reshape(self.batch_size, 1, 1)
            #target += np.dot(self.gamma,done) * target_next
            target += np.dot(self.gamma, done).dot(target_next)
            #target = np.reshape(15, 15)
            reshaped_target = np.max(target, axis=1)
            self.network.model.fit(state, reshaped_target, epochs=1, verbose=1)

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            print(f"Epsilon: {self.epsilon}")
        except Exception as e:
            print(f"Replay failed: {e}")

    def load(self, name):
        self.network.model.load_weights(name)

    def save(self, name):
        self.network.model.save_weights(name)

    def train(self, episodes, start_step, running_reward):
        steps_per_ep = 1000
        total_steps = start_step
        running_reward_tally = running_reward
        running_episode_mean = 0
        total_ep_means = 0
        for ep in range(episodes):
            state = self.env.reset()
            #state = np.reshape(state, [1, self.obs_space_flat_dim])
            done = False
            total_reward = 0
            steps = 0
            ep_steps = 0
            while True:
                action = self.act(state)
                #print(f"Action: {action}")
                next_state, reward, done, _ = self.env.step(action)
                #next_state = np.reshape(next_state, [1, self.obs_space_flat_dim])
                self.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                steps += 1
                ep_steps += 1
                if done:
                    total_steps += steps
                    steps = 0
                    running_reward_tally += total_reward
                    episode_mean = total_reward/ep_steps
                    total_ep_means += episode_mean
                    running_episode_mean += episode_mean
                    print(f"Ep: {ep}, final_score: {reward}, ep mean {episode_mean}, running mean: {running_reward_tally/total_steps} ")
                    print(f"total steps: {total_steps}, ep over ep mean: {total_ep_means/(ep + 1)}")
                    break
            self.replay()
        return total_steps, running_reward_tally
