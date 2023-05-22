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
from scenarios.SC2MiniMapScenario import SC2MiniMapScenario, MoveToBeaconScenario, DefeatRoachesScenario

with closing(create_connection("ws://127.0.0.1:5000/sc2api")) as websocket:
    scenario = SC2MiniMapScenario()
    env = SC2SyncEnv(websocket, scenario, 8)
    actions_n = 0
    n_games = 10000
    batch_size = 50

    if isinstance(env.action_space, gym.spaces.MultiDiscrete):
        actions_n = env.action_space.nvec[0]
    else:
        actions_n = env.action_space.n

    model = DQNAgent(env.observation_space.shape, actions_n, env)

    while True:
        try:
            model.learn(batch_size=batch_size, n_games=1000)
        except Exception as e:
            env.reset()
            continue
