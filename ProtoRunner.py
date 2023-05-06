from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from envs.SC2SyncEnv import SC2SyncEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C
from absl import flags
from contextlib import closing
from time import sleep
import websocket
from websocket import create_connection
FLAGS = flags.FLAGS
FLAGS([''])



with closing(create_connection("ws://127.0.0.1:5000/sc2api")) as websocket:
    checkpoint_callback = CheckpointCallback(save_freq=5000, save_path="./logs/raw_agent_runner/a2c",
                                             name_prefix="rl_model")
    env = SC2SyncEnv(websocket)
    env = DummyVecEnv([lambda: Monitor(env)])
    eval_callback = EvalCallback(env, best_model_save_path="./logs/raw_agent_runner/a2c/",
                                 log_path="./logs/raw_agent_runner/a2c/", eval_freq=10,
                                 deterministic=True, render=False,)

    model = A2C('MlpPolicy', env, learning_rate=0.001, gamma=0.85, verbose=1)
    #model = A2C.load("a2c_sc2_dbz")
    #model = A2C.load("logs/raw_agent_runner/a2c/best_model")
    model.env = env
    #env.reset()
    while True:
        try:
            model.learn(total_timesteps=100, callback=[checkpoint_callback, eval_callback])
        except Exception as e:
            continue
            env.reset()

    #model.learn(total_timesteps=100000)
    #model.save("a2c_sc2_dbz")
    #vec_env = model.get_env()

    #while True:
    #    obs = env.reset()
    #    try:
    #        action, _states = model.predict(obs)
    #        obs, rewards, dones, info = vec_env.step(action)

    #    except Exception as e:
    #        print(f"Error {e}")
