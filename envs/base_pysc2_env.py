from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from .base_gym_env import BaseGymEnv


class BasePysc2Env(BaseGymEnv):
    """
    Currently, this inherits from BaseGymEnv
    for convenience, but ultimately, it will
    only depend upon pysc2. A gym-based
    pysc2 agent would probably inherit
    from this.
    """
    metadata = {'render.modes': ['human']}

    default_settings = {
        'map_name': "DefeatZerglingsAndBanelings",
        'players': [sc2_env.Agent(sc2_env.Race.terran),
                    sc2_env.Bot(sc2_env.Race.zerg, sc2_env.Difficulty.very_easy)],
        'agent_interface_format': features.AgentInterfaceFormat(
            action_space=actions.ActionSpace.RAW,
            use_raw_units=True,
            raw_resolution=64),
        'realtime': False,
        'step_mul': 8
    }

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        args = {**self.default_settings, **self.kwargs}
        self.env = sc2_env.SC2Env(**args)
        #self.action_space = spaces.Discrete(56)
        #self.observation_space = spaces.Box(
        #    low=0,
        #    high=64,
        #    shape=(19, 3)
        #)

    def reset(self):
        pass

    def step(self, obs):
        pass