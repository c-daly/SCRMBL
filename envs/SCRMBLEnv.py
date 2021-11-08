"""
This is currently the only environment
supporting full games of starcraft,
but it will one day simply be the
base for all starcraft envs
"""
from pysc2.env import sc2_env
from pysc2.lib import actions, features
import numpy as np
import random
import logging
from envs import base_env

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fh = logging.FileHandler("botlog")
fh.setLevel(logging.INFO)
logger.addHandler(fh)


class SCRMBLEnv(base_env.BaseEnv):
    metadata = {'render.modes': ['human']}

    # TODO: send as command line args with defaults from
    #       the runner
    default_settings = {
        # 'map_name': "DefeatZerglingsAndBanelings",
        'map_name': "Simple64",
        'players': [sc2_env.Agent(sc2_env.Race.terran),
                    sc2_env.Bot(sc2_env.Race.zerg, sc2_env.Difficulty.medium)],
        'agent_interface_format': features.AgentInterfaceFormat(
            action_space=actions.ActionSpace.RAW,
            use_raw_units=True,
            raw_resolution=64),
        'realtime': False,
        'step_mul': 8
    }

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.env = sc2_env.SC2Env(**SCRMBLEnv.default_settings)
        self.marines = []
        self.barracks = []
        self.cc = None
        self.mineral_patches = []
        self.action_space = None
        self.observation_space = None

    def step(self, action_queue):
        translated_actions = self.translate_scrmbl_actions_to_pysc2(action_queue)
        return self.apply_actions_to_env(action_queue)

    def translate_scrmbl_actions_to_pysc2(self, action_queue):
        """
        Currently just a passthrough but I can imagine
        times where I'll want to be able to get at
        actions before they go the env.
        """
        return action_queue

    def apply_actions_to_env(self, action_queue):
        try:
            if len(action_queue) == 0 or action_queue is None:
                return self.env.step([actions.RAW_FUNCTIONS.no_op()])
            return self.env.step(action_queue)
        except Exception as ex:
            print(f"{action_queue}")
            raise ex

    def reset(self):
       raw_obs = self.env.reset()[0]
       return raw_obs, 0, raw_obs, {}

    def init_env(self):
        args = {**self.default_settings, **self.kwargs}
        self.env = sc2_env.SC2Env(**args)

    def close(self):
        if self.env is not None:
            self.env.close()
        super().close()

    def render(self, mode='human', close=False):
        pass
