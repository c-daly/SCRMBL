import gym


class BaseGymEnv(gym.Env):
    """
    This will be the entry point for
    all gym envs in the SCRMBL app.
    Should be able to just drop gym envs in.
    """
    def __init__(self):
        super(BaseGymEnv, self).__init__()

    def step(self, action):
        raise NotImplemented

    def reset(self):
        raise NotImplemented

    def init_env(self):
        raise NotImplemented

    def close(self):
        raise NotImplemented

    def render(self, mode='human', close=False):
        raise NotImplemented
