from stable_baselines3 import DQN, A2C
from stable_baselines3.common.env_util import make_atari_env

env = make_atari_env('BreakoutDeterministic-v4')

model = A2C('CnnPolicy', env, verbose=1, learning_rate=0.0009, tensorboard_log="./logs/progress_tensorboard/")
model.learn(total_timesteps=1000000)
model.save("deepq_breakout")
#while True:
#    model = DQN.load("deepq_breakout", env)
#    model.learn(total_timesteps=25000)
#    model.save("deepq_breakout")
#    env.reset()


#obs = env.reset()
#while True:
#    action, _states = model.predict(obs)
#    obs, rewards, dones, info = env.step(action)
#    env.render()