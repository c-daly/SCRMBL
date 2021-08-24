

class BaseEnv():
    """
    This will be the entry point for
    all gym envs in the SCRMBL app.
    Should be able to just drop gym envs in.
    """
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
