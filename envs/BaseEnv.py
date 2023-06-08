from abc import ABC

import gymnasium as gym


class BaseEnv(gym.Env, ABC):

    def __init__(self):
        self.envType = "Gym"
        self.num_envs = 1

    def reset(self):
        pass

    def step(self, action):
        pass

    def render(self):
        pass

    def close(self):
        pass
