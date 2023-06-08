from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from envs.SC2SyncEnvExtended import SC2SyncEnvExtended
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common import policies
from absl import flags
from contextlib import closing
from scenarios.DefeatZerglingsAndBanelingsScenario import DefeatZerglingsAndBanelingsScenario
from time import sleep
import websocket
from websocket import create_connection
FLAGS = flags.FLAGS
FLAGS([''])



with closing(create_connection("ws://127.0.0.1:5000/sc2api")) as websocket:
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path="./logs/raw_agent_runner/ppo",
                                             name_prefix="rl_model")
    scenario = DefeatZerglingsAndBanelingsScenario()
    env = SC2SyncEnvExtended(websocket, scenario, 64)
    #env = DummyVecEnv([lambda: Monitor(env)])
    #eval_callback = EvalCallback(env, best_model_save_path="./logs/raw_agent_runner/ppo/",
    #                                 log_path="./logs/raw_agent_runner/ppo/", eval_freq=1000,
        #                             deterministic=True, render=False, n_eval_episodes=3)

    model = PPO(policies.ActorCriticPolicy, env=env, learning_rate=0.001, gamma=0.99, verbose=1)
    #model = PPO.load("logs/raw_agent_runner/ppo/rl_model_63000_steps.zip")
    model.env = env
    while True:
        try:
            model.learn(total_timesteps=100000, callback=[checkpoint_callback])
        except Exception as e:
            env.reset()
            continue

    #while True:
    #    obs = env.reset()
    #    try:
    #        action, _states = model.predict(obs)
    #        obs, rewards, dones, info = env.step(action)

    #    except Exception as e:
    #        print(f"Error {e}")
