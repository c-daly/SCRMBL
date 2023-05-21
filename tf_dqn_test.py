#!/usr/bin/env python
# Deep Q Network (DQN) agent training script
# Chapter 3, TensorFlow 2 Reinforcement Learning Cookbook | Praveen Palanisamy

import argparse
from datetime import datetime
import os
import random
from collections import deque

import gym
import keras.layers
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam, Nadam

tf.keras.backend.set_floatx("float64")

parser = argparse.ArgumentParser(prog="TFRL-Cookbook-Ch3-DQN")
parser.add_argument("--env", default="CartPole-v0")
parser.add_argument("--lr", type=float, default=0.005)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--gamma", type=float, default=0.95)
parser.add_argument("--eps", type=float, default=1.0)
parser.add_argument("--eps_decay", type=float, default=0.995)
parser.add_argument("--eps_min", type=float, default=0.01)
parser.add_argument("--logdir", default="logs")

args = parser.parse_args()
logdir = os.path.join(
    args.logdir, parser.prog, args.env, datetime.now().strftime("%Y%m%d-%H%M%S")
)
print(f"Saving training logs to:{logdir}")
writer = tf.summary.create_file_writer(logdir)


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def store(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])

    def sample(self):
        sample = random.sample(self.buffer, args.batch_size)
        states, actions, rewards, next_states, done = map(np.asarray, zip(*sample))
        states = np.array(states).reshape(args.batch_size, -1)
        #next_states = np.array(next_states).reshape(args.batch_size, -1)
        return states, actions, rewards, next_states, done

    def size(self):
        return len(self.buffer)


class DQN:
    def __init__(self, input_shape, actions_n):
        self.input_shape = input_shape
        self.actions_n = actions_n
        self.action_dim = actions_n
        self.epsilon = 0.1

        self.model = self.nn_model()

    def nn_model(self):
        model = tf.keras.Sequential()
        model.add(Dense(32, activation='relu', input_shape=(2, 2)))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.actions_n, activation='linear'))
        model.compile(loss='mse', optimizer=Nadam(lr=0.001))

        return model


    def predict(self, state):
        return self.model.predict(state)

    def get_action(self, state):
        #state = np.reshape(state, [1, self.state_dim])
        #self.epsilon *= args.eps_decay
        #self.epsilon = max(self.epsilon, args.eps_min)
        q_value = self.predict(state[0])
        q_value = q_value[0]
        if np.random.random() < self.epsilon:
            action = random.randint(0, self.action_dim - 1)
            print(f"action/qvalue: {action}/{q_value}")
            return random.randint(0, self.action_dim - 1)
        action = np.argmax(q_value)
        print(f"action: {action}")
        return action

    def train(self, states, targets):
        self.model.fit(states, targets, epochs=1)


class Agent:
    def __init__(self, env):
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = 4

        self.model = DQN(self.env.observation_space.shape, self.action_dim)
        self.target_model = DQN(self.env.observation_space.shape, self.action_dim)
        self.update_target()

        self.buffer = ReplayBuffer()

    def update_target(self):
        weights = self.model.model.get_weights()
        self.target_model.model.set_weights(weights)

    def replay_experience(self):
        for _ in range(10):
            states, actions, rewards, next_states, done = self.buffer.sample()
            targets = self.model.predict(states)
            next_q_values = self.target_model.predict(next_states) #.max(axis=1)
            targets[range(args.batch_size), actions] = (
                rewards + (1 - done) * next_q_values * args.gamma
            )
            self.model.train(states, targets)

    def train(self, max_episodes=100000):
        with writer.as_default():  # Tensorboard logging
            for ep in range(max_episodes):
                done, episode_reward = False, 0
                observation = self.env.reset()
                print(f"Done: {done}")
                steps = 0
                while not done:
                    if steps > 1000:
                        done = True
                    steps += 1
                    #self.env.render()
                    action = self.model.get_action(observation)
                    if np.isscalar(action):
                        action = [action]
                    next_observation, reward, done, _ = self.env.step(action)
                    print(f"reward: {reward}")
                    self.buffer.store(
                        observation, action, reward, next_observation, done
                    )
                    episode_reward += reward
                    observation = next_observation
                if self.buffer.size() >= args.batch_size:
                    self.replay_experience()
                self.update_target()
                print(f"Episode#{ep} Reward:{episode_reward}")
                #tf.summary.scalar("episode_reward", episode_reward, step=ep)
                #print(f"ep reward: {episode_reward}")
                writer.flush()
    def learn(self, max_episdoes=1000):
        minibatch = self.repla
