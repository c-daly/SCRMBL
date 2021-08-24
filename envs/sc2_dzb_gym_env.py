from gym import spaces
from pysc2.lib import actions, units
import numpy as np
import random
from .base_pysc2_env import BasePysc2Env


class SC2DZBGymEnv(BasePysc2Env):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_space = spaces.Discrete(56)
        self.observation_space = spaces.Box(
            low=0,
            high=64,
            shape=(19, 3)
        )

    def step(self, action):
        raw_obs = self.take_action(action)
        reward = raw_obs.reward
        obs = self.get_derived_obs(raw_obs)
        return obs, reward, raw_obs.last(), {}

    def take_action(self, action):
        mapped_action = action % 6
        marine_index = (action % 9)
        try:
            marine = self.marines[max(marine_index, 0)]
        except:
            marine = None
        action_mapped = actions.RAW_FUNCTIONS.no_op()
        if mapped_action == 0:
            action_mapped = self.move_up(marine)
        elif mapped_action == 1:
            action_mapped = self.move_right(marine)
        elif mapped_action == 2:
            action_mapped = self.move_down(marine)
        elif mapped_action == 3:
            action_mapped = self.move_left(marine)
        elif mapped_action == 4:
            action_mapped = self.attack(marine)
        try:
            raw_obs = self.env.step([action_mapped])
            return raw_obs[0]
        except:
            action_mapped = actions.RAW_FUNCTIONS.no_op()
            raw_obs = self.env.step([action_mapped])
            return raw_obs[0]

    def move_up(self, marine):
        try:
            new_pos = [marine.x, marine.y - 2]
            return actions.RAW_FUNCTIONS.Move_pt("now", marine.tag, new_pos)
        except:
            return actions.RAW_FUNCTIONS.no_op()

    def move_down(self, marine):
        try:
            new_pos = [marine.x, marine.y + 2]
            return actions.RAW_FUNCTIONS.Move_pt("now", marine.tag, new_pos)
        except:
            return actions.RAW_FUNCTIONS.no_op()

    def move_left(self, marine):
        try:
            new_pos = [marine.x - 2, marine.y]
            return actions.RAW_FUNCTIONS.Move_pt("now", marine.tag, new_pos)
        except:
            return actions.RAW_FUNCTIONS.no_op()

    def move_right(self, marine):
        try:
            new_pos = [marine.x + 2, marine.y]
            return actions.RAW_FUNCTIONS.Move_pt("now", marine.tag, new_pos)
        except:
            return actions.RAW_FUNCTIONS.no_op()

    def attack(self, marine):
        try:
            enemy = random.choice(self.enemy_units)
            return actions.RAW_FUNCTIONS.Attack_pt("now", marine.tag, (enemy.x, enemy.y))
        except:
            return actions.RAW_FUNCTIONS.no_op()

    def attack_enemies(self):
        if len(self.marines) > 0:
            tags = [marine.tag for marine in self.marines]
            if len(self.enemy_units) > 0:
                enemy = random.choice(self.enemy_units)
                return actions.RAW_FUNCTIONS.Attack_pt("now", tags, (enemy.x, enemy.y))
            attack_xy = (random.randint(1, 83), random.randint(1, 83))
            return actions.RAW_FUNCTIONS.Attack_pt("now", tags, attack_xy)
        else:
            return actions.RAW_FUNCTIONS.no_op()

    def reset(self):
        if self.env is None:
            self.init_env()
        self.marines = []
        self.zerglings = []
        self.banelings = []
        raw_obs = self.env.reset()[0]
        return self.get_derived_obs(raw_obs)

    def get_derived_obs(self, raw_obs):
        obs = np.zeros((19, 3), dtype=np.uint8)
        marines = self.get_units_by_type(raw_obs, units.Terran.Marine, 1)
        zerglings = self.get_units_by_type(raw_obs, units.Zerg.Zergling, 4)
        banelings = self.get_units_by_type(raw_obs, units.Zerg.Baneling, 4)
        self.marines = []
        self.banelings = []
        self.zerglings = []
        for i, m in enumerate(marines):
            self.marines.append(m)
            obs[i] = np.array([m.x, m.y, m[2]])
        for i, b in enumerate(banelings):
            self.banelings.append(b)
            obs[i + 9] = np.array([b.x, b.y, b[2]])
        for i, z in enumerate(zerglings):
            self.zerglings.append(z)
            obs[i + 13] = np.array([z.x, z.y, z[2]])
        return obs

    def get_enemy_units(self, obs):
        return [unit for unit in obs.observation.raw_units if unit.alliance == 4]

    def get_units_by_type(self, obs, unit_type, player_relative=0):
        """
        NONE = 0
        SELF = 1
        ALLY = 2
        NEUTRAL = 3
        ENEMY = 4
        """
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.alliance == player_relative]

    def get_distances(self, units, xy):
        units_xy = [(unit.x, unit.y) for unit in units]
        return np.linalg.norm(np.array(units_xy) - np.array(xy), axis=1)
