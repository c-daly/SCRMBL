from abc import ABC

import gymnasium as gym


class BaseEnv(gym.Env, ABC):

    def __init__(self):
        self.envType = "Gym"

    def reset(selfself):
        pass

    def step(self, action):
        pass

    def render(self):
        pass

    def close(self):
        pass
