import numpy as np
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.optimizers import Adam, schedules
from tensorflow import keras
from collections import deque
from envs.SC2SyncEnv import SC2SyncEnv
from contextlib import closing
import gym
from websocket import create_connection
from scenarios.SC2MiniMapScenario import SC2MiniMapScenario, MoveToBeaconScenario, DefeatRoachesScenario
from algos.DQN import DQNAgent

with closing(create_connection("ws://127.0.0.1:5000/sc2api")) as websocket:
    scenario = SC2MiniMapScenario()
    env = SC2SyncEnv(websocket, scenario, 4)
    actions_n = 0
    n_games = 10000
    batch_size = 1024
    capacity = 1024

    if isinstance(env.action_space, gym.spaces.MultiDiscrete):
        actions_n = env.action_space.nvec[0]
    else:
        actions_n = env.action_space.n

    model = DQNAgent(env, env.observation_space, env.action_space, batch_size, capacity)
    #scenario.model = keras.models.load_model("mtb.dqn.h5")
    #model.network.model = scenario.model
    #model.network.model = scenario.model
    scenario.model = model.network
    start_step = 0
    running_reward = 0
    num_episodes = 100
    ep = 0
    while True:
        try:
            start_step, running_reward = model.train(num_episodes, start_step, running_reward)
            ep += num_episodes
            model.network.model.save("mtb.dqn.h5")
        except Exception as e:
            env.reset()
            continue
    print(f"Eps: {ep}, Steps: {start_step}")
