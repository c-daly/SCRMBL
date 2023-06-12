import numpy as np
import random
from networks.DeepQNetwork import DeepQNetwork
from networks.CNN import CNN, CNN2
from collections import deque
import gym

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
        self.episodes = 0
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
        self.epsilon = .5  # Exploration
        self.epsilon_min = 0.35
        self.epsilon_decay = 0.995
        self.batch_size = batch_size
        self.capacity = capacity

        #self.network = DeepQNetwork(self.obs_space_flat_dim, self.action_space_flat_dim)
        #self.network = CNN(self.obs_space, self.action_space_flat_dim)
        self.network = CNN2(self.obs_space, self.action_space_flat_dim)


    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def act(self, state):
        if isinstance(self.env.action_space, gym.spaces.MultiDiscrete):
            size = (1, self.action_space.shape[0], self.action_space.nvec[0])
        else:
            size = (1, self.action_space_flat_dim)
            random_choice = np.random.choice(4, size=(1,))

        if np.random.rand() <= self.epsilon:
            if isinstance(self.env.action_space, gym.spaces.Discrete):
                return random_choice
            return np.random.choice(self.action_space.nvec[0], size=(len(self.action_space.nvec)))
        reshaped_state = np.expand_dims(state, axis=0)
        try:
            net_result = self.network.model.predict(reshaped_state, verbose=0)
        except Exception as e:
            print(f"Predict Exception: {e}")
            return np.random.choice(self.action_space.nvec[0], size=(len(self.action_space.nvec)))
        action = net_result.reshape(size)

        if isinstance(self.env.action_space, gym.spaces.MultiDiscrete):
            action = np.argmax(action[0], axis=1)
        else:
            action = np.argmax(action)
        return action

    def lr_schedule(self):
        # Learning rate schedule
        #lr = 0.1 * (0.9 ** self.episodes)
        lr = 0.00001
        return lr

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        try:
            result = self.memory.sample(self.batch_size)
            state = np.array([a[0] for a in result])
            state = state.reshape(self.batch_size, 64, 64, 3)
            next_state = np.array([a[3] for a in result])
            next_state = next_state.reshape(self.batch_size, 64, 64, 3)
            done = [int(a[4]) for a in result]
            reward = [a[2] for a in result]
            target = self.network.model.predict(state, verbose=0)
            target_next = self.network.model.predict(next_state, verbose=0)
            target += np.dot(self.gamma, done).dot(target_next)
            reshaped_target = np.max(target, axis=1)

            self.network.model.optimizer.learning_rate.assign(self.lr_schedule())

            self.network.model.fit(state, reshaped_target, epochs=1, verbose=1, batch_size=self.batch_size)

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            print(f"Epsilon: {self.epsilon}")
            print(f"Learning rate: {self.network.model.optimizer.learning_rate.value()}")
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
            done = False
            total_reward = 0
            steps = 0
            ep_steps = 0
            while True:
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                steps += 1
                ep_steps += 1
                if done:
                    self.episodes += 1
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
