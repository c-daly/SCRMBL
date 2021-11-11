import gym
import numpy as np
import random


#env = gym.make("Taxi-v3").env
env = gym.make("FrozenLake-v0").env

q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1

# For plotting metrics
all_epochs = []
all_penalties = []

for i in range(1, 10000001):
    state = env.reset()

    epochs, penalties, reward, = 0, 0, 0
    done = False

    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore action space
        else:
            action = np.argmax(q_table[state])  # Exploit learned values

        next_state, reward, done, info = env.step(action)

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        if reward == -10:
            penalties += 1

        state = next_state
        epochs += 1

    if i % 100 == 0:
        print(f"Episode: {i}, reward: {reward}")

print("Training finished.\n")
"""Evaluate agent's performance after Q-learning"""

total_epochs, total_penalties, total_rewards = 0, 0, 0
episodes = 100

for _ in range(episodes):
    state = env.reset()
    epochs, penalties, reward, rewards = 0, 0, 0, 0

    done = False

    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action)
        print(f"Reward: {reward}")
        rewards += reward
        epochs += 1

    total_penalties += penalties
    total_epochs += epochs
    total_rewards += rewards

print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average rewards per episode: {total_rewards}")