import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import numpy as np
import gym
from skimage.color import rgb2gray
from skimage.transform import resize
from tensorflow.keras.layers import Conv2D, Flatten
from numpy import random
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from envs.SC2SyncEnv import SC2SyncEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C, PPO
from absl import flags
from contextlib import closing
from time import sleep
import websocket
from websocket import create_connection

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

#class ReplayBuffer:
#    def __init__(self, capacity):
#        self.capacity = capacity
#        self.buffer = []
#        self.memory = np.array([])
#        self.position = 0

#    def push(self, state, action, reward, next_state, done):
#        if len(self.buffer) < self.capacity:
#            self.buffer.append(None)
#        self.buffer[self.position] = (state, action, reward, next_state, done)
#        self.position = (self.position + 1) % self.capacity

    #def sample(self, batch_size):
    #    batch = np.random.choice(len(self.buffer), batch_size, replace=False)
    #    state, action, reward, next_state, done = zip(*[self.buffer[idx] for idx in batch])
    #    return np.stack(state), action, reward, np.stack(next_state), done

    #def sample(self, batch_size):
    #    idx = np.random.choice(len(self.buffer), batch_size, replace=False)
    #    batch = np.array(self.buffer)[idx]
#    #    state = np.stack(batch[:, 0])
    #    action = batch[:, 1]
    #    reward = batch[:, 2]
    #    next_state = np.stack(batch[:, 3])
    #    done = batch[:, 4]

    #    return state, action, reward, next_state, done

#    def sample(self, batch_size):
#        indices = np.random.randint(len(self.buffer), size=batch_size)
#        minibatch = [self.buffer[i] for i in indices]
#        state, action, reward, next_state, done = map(np.array, zip(*minibatch))
#
#        # Convert to appropriate data types
#        state = state.astype(np.float32)
#        action = action.astype(np.int32)
#        reward = reward.astype(np.float32)
#        next_state = next_state.astype(np.float32)
#        done = done.astype(np.bool)

#        return state, action, reward, next_state, done

#    def __len__(self):
#        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_space, action_space, epsilon=0.9, gamma=0.95, lr=1e-3):
        self.state_shape = state_space
        self.action_space = action_space
        self.n_actions = action_space
        #if not isinstance(gym.spaces.multi_discrete.MultiDiscrete, action_space):
        #    self.action_space.n
        self.epsilon = epsilon
        self.gamma = gamma
        self.model = self.create_model(lr)

    #def create_md_model(self):
    #    # Input layer
    #    input_layer = layers.Input(shape=m_observation_space,))

        # Hidden layers
    #    hidden_layer1 = layers.Dense(64, activation='relu')(input_layer)
    #    hidden_layer2 = layers.Dense(64, activation='relu')(hidden_layer1)

        # Output layers
    #    output_layers = [layers.Dense(4, activation='linear')(hidden_layer2) for _ in range(9)]

        # Create model
    #    model = tf.keras.Model(inputs=input_layer, outputs=output_layers)

    def create_model(self, lr):

        model = Sequential()
        model.add(Input(self.state_shape))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.n_actions, activation='linear'))
        model.compile(optimizer=Adam(lr=lr), loss='mse')
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
        return model

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.n_actions)

        q_values = self.model.predict(state)
        print(f"q value: {np.max(q_values[0])}")
        return np.argmax(q_values[0]) % self.n_actions

    def train(self, replay_buffer, batch_size=10):
        #if len(replay_buffer) < batch_size:
        #    return
        if replay_buffer.size < batch_size:
            return
        try:
            state, action, reward, next_state, done = replay_buffer.sample(batch_size)
            action = np.array(action)
            reward = np.array(reward)
            done = np.array(done)
            print(f"action/reward: {action}/{reward}")
            #state = np.vstack(state)
            #state = state[np.newaxis, :]
            target = self.model.predict(state)
            target_next = self.model.predict(next_state)

            for idx in range(batch_size):
                #if not isinstance(target[0], int):
                #    target = target[0]
                if idx >= len(target):
                    break
                if done[idx]:
                    target[idx][action[idx]] = reward[idx] * .0001
                else:
                    target_reward = (reward[idx] + self.gamma * np.max(target_next[idx])) * .0001
                    #if not np.isscalar(target_reward):
                    #    target_reward = target_reward[0]
                    #if np.isscalar(target[0]):
                    #    target = [target]
                    #    if idx > 0:
                    #        break
                    target[idx][0][action[idx]] = target_reward
            #target[np.arange(batch_size), action] = reward + self.gamma * np.max(target_next, axis=1)
            #target[done, action[done]] = reward[done]

            self.model.fit(state, target, epochs=1, verbose=0)
        except Exception as e:
            print(f"error: {e}")

#def main():
#    env = gym.make("CartPole-v1")
#    n_games = 400
#    batch_size = 64
#    replay_buffer = ReplayBuffer(capacity=5000)
#
#    agent = DQNAgent(env.observation_space.shape, env.action_space.n)
#
#    for i in range(n_games):
#        state = env.reset()
#        state = preprocess(state)
#        state_stack = np.stack([state] * 4, axis=2)
#        #state = np.reshape(state, [1, agent.state_shape[0]])
#        total_reward = 0
#
#        while True:
#            action = agent.act(state)
#            next_state, reward, done, _ = env.step(action)
#            env.render()
#            next_state = np.reshape(next_state, [1, agent.state_shape[0]])
#
#            replay_buffer.push(state, action, reward, next_state, done)
#            state = next_state
#            total_reward += reward
#
#            if done:
#                break
#
#            agent.train(replay_buffer, batch_size)
#
#        if i % 10 == 0:
#            print(f"Episode: {i}, Reward: {total_reward}")
#
#        # Epsilon decay
#        if agent.epsilon > 0.01:
#            agent.epsilon *= 0.995
#

def main():
    env = gym.make('SpaceInvaders-v4', render_mode='human')
    n_games = 400
    batch_size = 150
    replay_buffer = ReplayBuffer(capacity=5000)

    agent = DQNAgent(env.observation_space.shape, env.action_space.n)

    for i in range(n_games):
        state = env.reset()
        total_reward = 0

        while True:
            action = agent.act(state_stack[np.newaxis, :])  # Reshape to (1, 84, 84, 4)
            next_state, reward, done, _ = env.step(action)
            print(f"action/reward: {action}/{reward}")
            next_state = np.append(state[:, :, 1:], np.expand_dims(next_state, 2), axis=2)

            replay_buffer.push(state, action, reward, next_state, done)
            state_stack = next_state
            total_reward += reward

            if done:
                break

            agent.train(replay_buffer, batch_size)

        if i % 10 == 0:
            print(f"Episode: {i}, Reward: {total_reward}")

        # Epsilon decay
        if agent.epsilon > 0.01:
            agent.epsilon *= 0.995


#if __name__ == "__main__":
#    main()


with closing(create_connection("ws://127.0.0.1:5000/sc2api")) as websocket:
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path="../logs/raw_agent_runner/ppo",
                                             name_prefix="rl_model")
    env = SC2SyncEnv(websocket)
    env = DummyVecEnv([lambda: Monitor(env)])
    actions_n = 0
    n_games = 400
    batch_size = 5
    replay_buffer = ReplayBuffer(capacity=10)

    if isinstance(env.action_space, gym.spaces.MultiDiscrete):
        actions_n = env.action_space.nvec[0]
    else:
        actions_n = env.action_space.n
    #agent = DQNAgent(env.observation_space.shape, env.action_space.n)
    agent = DQNAgent(env.observation_space.shape, env.action_space.n)

    for i in range(n_games):
        state = env.reset()
        state = state.reshape(state.shape[1:])
        total_reward = 0

        while True:
            action = agent.act(state[np.newaxis, :])  # Reshape to (1, 84, 84, 4)
            next_state, reward, done, _ = env.step([action])
            print(f"action/reward: {action}/{reward}")
            #next_state = np.append(state[:, :, 1:], np.expand_dims(next_state, 2), axis=2)
            next_state = next_state.reshape(next_state.shape[1:])
            replay_buffer.push(state, action, reward, next_state, done)
            state_stack = next_state
            total_reward += reward

            if done:
                break

            agent.train(replay_buffer, batch_size)

        if i % 10 == 0:
            print(f"Episode: {i}, Reward: {total_reward}")

        # Epsilon decay
        if agent.epsilon > 0.01:
            agent.epsilon *= 0.995

    #env.reset()
    while True:
        try:
            model.learn(total_timesteps=100000, callback=[eval_callback, checkpoint_callback])
        except Exception as e:
            env.reset()
            continue

