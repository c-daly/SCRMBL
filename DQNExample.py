from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from envs.SC2SyncEnv import SC2SyncEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from algos.DQN import DQNAgent
from stable_baselines3 import A2C, PPO
from absl import flags
from contextlib import closing
from time import sleep
import websocket
import gym
from websocket import create_connection
from scenarios.MoveToBeaconScenario import MoveToBeaconScenario
from tensorflow import keras
from networks.DeepQNetwork import DeepQNetwork

with closing(create_connection("ws://127.0.0.1:5000/sc2api")) as websocket:
    scenario = MoveToBeaconScenario()
    env = SC2SyncEnv(websocket, scenario, 16)
    actions_n = 0
    n_games = 10000
    batch_size = 2
    capacity = 4

    if isinstance(env.action_space, gym.spaces.MultiDiscrete):
        actions_n = env.action_space.nvec[0]
    else:
        actions_n = env.action_space.n

    model = DQNAgent(env, env.observation_space, env.action_space, batch_size, capacity)
    #scenario.model = keras.models.load_model("mtb.dqn.h5")
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
            raise e
            env.reset()
            continue
    print(f"Eps: {ep}, Steps: {start_step}")
