from algos.qlearning import qlearning
from stable_baselines3.common.env_util import make_atari_env
import gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from envs.SC2SyncEnv import SC2SyncEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C, PPO

#env = gym.make('Taxi-v3') #, is_slippery=False)
env = gym.make("SpaceInvaders-v4", render_mode="human")
#model = qlearning(env) #, render="human")

#model = qlearning(env=env)

model = PPO('CnnPolicy', env=env, learning_rate=0.01, gamma=0.99, verbose=1)
#while True:
#    try:
#model.learn()
#    except Exception as e:
#        raise e

name_prefix = "atari_model"
env = DummyVecEnv([lambda: Monitor(env)])
checkpoint_callback = CheckpointCallback(save_freq=1000, save_path="../logs/raw_agent_runner/ppo",
                                         name_prefix="rl_model")
eval_callback = EvalCallback(env, best_model_save_path="../logs/raw_agent_runner/ppo_atari/",
                             log_path="../logs/raw_agent_runner/ppo/", eval_freq=1000,
                             deterministic=True, render=False)

# env.reset()
while True:
    try:
        model.learn(total_timesteps=1000, callback=[eval_callback, checkpoint_callback])
        env.reset()
    except Exception as e:
        print(f"Exception: {e}")
        env.reset()
        continue

# watch trained agent
#state = model.env.reset()
#rewards = 0

#for i in range(10):
#    env.reset()
#    done = False
#    while not done:

#        action = model.choose_action(state)
#        new_state, reward, done, info = model.env.step(action)
#        env.render()
#        state = new_state
#        rewards += reward
#        if done:
#            print(f"reward {reward}")

