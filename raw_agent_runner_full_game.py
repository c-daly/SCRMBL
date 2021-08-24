from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from envs.sc2_gym_env import SC2GymEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C
from absl import flags

FLAGS = flags.FLAGS
FLAGS([''])

checkpoint_callback = CheckpointCallback(save_freq=5000, save_path="./logs/raw_agent_runner/a2c",
                                         name_prefix="rl_model")

env = DummyVecEnv([lambda: Monitor(SC2GymEnv())])
eval_callback = EvalCallback(env, best_model_save_path="./logs/raw_agent_runner/a2c/",
                             log_path="./logs/raw_agent_runner/a2c/", eval_freq=10000,
                             deterministic=True, render=False,)

#model = A2C('MlpPolicy', env, learning_rate=0.0007, verbose=1)
model = A2C.load('a2c_sc2_full')
model.env = env

model.learn(total_timesteps=100000, callback=[checkpoint_callback, eval_callback])
model.save("a2c_sc2_full")