import pickle
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from envs.SC2SyncEnv import SC2SyncEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C, PPO
from absl import flags
from contextlib import closing
from algos.qlearning import qlearning
from time import sleep
import websocket
from websocket import create_connection
import numpy as np
FLAGS = flags.FLAGS
FLAGS([''])



with closing(create_connection("ws://127.0.0.1:5000/sc2api")) as websocket:
    checkpoint_callback = CheckpointCallback(save_freq=5000, save_path="./logs/raw_agent_runner/qlearn",
                                             name_prefix="rl_model")
    env = SC2SyncEnv(websocket)
    env = DummyVecEnv([lambda: Monitor(env)])
    eval_callback = EvalCallback(env, best_model_save_path="./logs/raw_agent_runner/qlearn/",
                                 log_path="../logs/raw_agent_runner/ppo/", eval_freq=64,
                                 deterministic=True, render=False, )

    #model = PPO('MlpPolicy', env=env, learning_rate=0.0001, gamma=0.9, verbose=1)
    #model = A2C.load("a2c_sc2_dbz")
    #model = A2C.load("logs/raw_agent_runner/a2c/best_model")
    #model = A2C.load("logs/raw_agent_runner_new_obs/a2c/rl_model_1000_steps")
    #model = PPO.load("logs/raw_agent_runner/ppo/rl_model_848_steps")
    qtable = None
    actions_by_state = None
    #with open('qtable', 'rb') as fp:
    #    qtable = pickle.load(fp)
    #with open('actions_by_state', 'rb') as fp:
    #    actions_by_state = pickle.load(fp)
    model = qlearning(env, render=None, callbacks=[eval_callback, checkpoint_callback]) #, saves=[qtable, actions_by_state])
    #model.env = env
    #env.reset()
    while True:
        try:
            model.learn()
        except Exception as e:
            raise e
            env.reset()
            continue

    #model.learn(total_timesteps=100000)
    #model.save("a2c_sc2_dbz")
    #vec_env = model.get_env()

    #while True:
    #    obs = env.reset()
    #    try:
    #        action, _states = model.predict(obs)
    #        obs, rewards, dones, info = env.step(action)

    #    except Exception as e:
    #        print(f"Error {e}")
