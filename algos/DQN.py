import numpy as np
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input, Conv2D
from tensorflow.keras.optimizers import Adam, schedules
from tensorflow import keras
from collections import deque
from envs.SC2SyncEnv import SC2SyncEnv
from contextlib import closing
import gym
from websocket import create_connection
from scenarios.SC2MiniMapScenario import SC2MiniMapScenario, MoveToBeaconScenario, DefeatRoachesScenario

# Define the Q-Network
class DQNetwork:
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
        self.model.compile(loss='mse', optimizer=Adam(learning_rate=lr_schedule))

class MultiDiscreteDQNetwork:
    def __init__(self, input_shape, value_max, num_values):
        lr_schedule = schedules.ExponentialDecay(
            initial_learning_rate=.01,
            decay_steps=50,
            decay_rate=0.5)

        # Input layer
        input_layer = Input(shape=input_shape.shape)

        # Hidden layers
        hidden_layer1 = Dense(64, activation='relu')(input_layer)
        hidden_layer2 = Dense(64, activation='relu')(hidden_layer1)

        # Output layers
        output_layers = [Dense(value_max, activation='linear')(hidden_layer2) for _ in range(num_values)]

        # Create model
        model = keras.Model(inputs=input_layer, outputs=output_layers)
        model.compile(optimizer=Adam(lr=0.0001), loss='mse')

class ConvDQNetwork:
    def __init__(self, obs_space, action_space):
        # Network defined by the Deepmind paper
        lr_schedule = schedules.ExponentialDecay(
            initial_learning_rate=.01,
            decay_steps=50,
            decay_rate=0.5)

        self.model = Sequential()
        self.model.add(Input(shape=(1, 64, 64, 3)))

        # Convolutions on the frames on the screen
        self.model.add(Conv2D(32, 8, strides=4, activation="relu"))
        self.model.add(Conv2D(64, 4, strides=2, activation="relu"))
        self.model.add(Conv2D(64, 3, strides=1, activation="relu"))

        self.model.add(Flatten())

        self.model.add(Dense(512, activation="relu"))
        self.model.add(Dense(action_space, activation="linear"))
        self.model.compile(loss='mse', optimizer=Adam(learning_rate=lr_schedule))

# Define Replay Memory
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
        #return np.random.choice(self.memory, batch_size)

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
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.95
        self.batch_size = batch_size
        self.capacity = capacity

        self.network = self.build_multidiscrete_network()
        #self.network = self.build_convolutional_network()

    def build_standard_network(self):
        return DQNetwork(self.obs_space_flat_dim, self.action_space_flat_dim)

    def build_convolutional_network(self):
        return ConvDQNetwork(self.obs_space, self.action_space_flat_dim)

    def build_multidiscrete_network(self):
        return MultiDiscreteDQNetwork(self.obs_space, self.action_space.nvec[0], len(self.action_space.nvec))

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
        net_result = self.network.model.predict(state, verbose=0)
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

        result = self.memory.sample(self.batch_size)
        state = np.array([a[0] for a in result])
        #state = state.reshape(self.batch_size, 1, self.obs_space_flat_dim)
        next_state = np.array([a[3] for a in result])
        #next_state = next_state.reshape(self.batch_size, 1, self.obs_space_flat_dim)
        done = [a[4] for a in result]
        done_ints = np.zeros(np.shape(done))
        reward = [a[2] for a in result]
        target = self.network.model.predict(state, verbose=0)
        target_next = self.network.model.predict(next_state, verbose=0)

        target += self.gamma * target_next * done_ints.reshape(self.batch_size, 1, 1)
        #target = reward
        #if not done:
        #target += self.gamma * target_f
        #    target_f = target
        #self.network.model.fit(state, target_next, epochs=1, verbose=0)
        self.network.model.fit(state, target, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            print(f"New epsilon: {self.epsilon}")

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
            #for i in range(steps_per_ep):
                action = self.act(state)
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
                    print(f"Learning rate: {self.network.model.optimizer.learning_rate.numpy()}")
                    break
            self.replay()
        return total_steps, running_reward_tally

#with closing(create_connection("ws://127.0.0.1:5000/sc2api")) as websocket:
#    scenario = SC2MiniMapScenario()
#    #scenario.model = keras.models.load_model("dqn.h5")
#
#    env = SC2SyncEnv(websocket, scenario, 1)
#    actions_n = 0
#    n_games = 10000
#    batch_size = 128
#    capacity = 150
#
#    if isinstance(env.action_space, gym.spaces.MultiDiscrete):
#        actions_n = env.action_space.nvec[0]
#    else:
#        actions_n = env.action_space.n
#
#    model = DQNAgent(env, env.observation_space.shape, env.action_space.shape, batch_size, capacity)
#    #model.network.model = scenario.model
#    scenario.model = model.network
#    start_step = 0
#    running_reward = 0
#    num_episodes = 100
#    ep = 0
#    while True:
#        try:
#            start_step, running_reward = model.train(num_episodes, start_step, running_reward)
#            ep += num_episodes
#            start_step = model.steps
#            model.network.model.save("dqn.h5")
#        except Exception as e:
#            env.reset()
#            continue
#    print(f"Eps: {ep}, Steps: {start_step}")
