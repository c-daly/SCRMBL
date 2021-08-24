from gym import spaces
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
import numpy as np
import random
from .base_gym_env import  BaseGymEnv


class SC2GymEnv(BaseGymEnv):
    metadata = {'render.modes': ['human']}

    default_settings = {
        # 'map_name': "DefeatZerglingsAndBanelings",
        'map_name': "Simple64",
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
        super(SC2GymEnv, self).__init__()
        self.kwargs = kwargs
        self.env = None
        self.marines = []
        self.mineral_patches = []
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(
            low=0,
            high=64,
            shape=(500, 4)
        )

    def step(self, action):
        raw_obs = self.take_action(action)  # take safe action
        reward = raw_obs.reward  # get reward from the env
        obs = self.get_derived_obs(raw_obs)  # get derived observation
        return obs, reward, raw_obs.last(), {}

    def take_action(self, action):
        # map value to action
        mapped_action = action % 9
        if mapped_action == 0:
            action_mapped = self.harvest_minerals()
        elif mapped_action == 1:
            action_mapped = self.build_supply_depot()
        elif mapped_action == 2:
            action_mapped = self.train_scv()
        elif mapped_action == 3:
            action_mapped = self.build_barracks()
        elif mapped_action == 4:
            action_mapped = self.train_marine()
        elif mapped_action == 5:
            action_mapped = self.attack_enemies()
        elif mapped_action == 6:
            action_mapped = self.scvs_attack()
        elif mapped_action == 7:
            action_mapped = self.all_scvs_harvest_minerals()
        else:
            action_mapped = actions.RAW_FUNCTIONS.no_op()
        try:
            raw_obs = self.env.step([action_mapped])
            return raw_obs[0]
        except:
            action_mapped = actions.RAW_FUNCTIONS.no_op()
            raw_obs = self.env.step([action_mapped])
            return raw_obs[0]

    def scvs_attack(self):
        if len(self.enemy_units) > 0:
            tags = []
            if len(self.scvs) > 0:
                tags = [scv.tag for scv in self.scvs]
            enemy = random.choice(self.enemy_units)
            return actions.RAW_FUNCTIONS.Attack_pt("now", tags, (enemy.x, enemy.y))
        else:
            return actions.RAW_FUNCTIONS.no_op()

    def attack_enemies(self):
        if len(self.marines) > 0:
            tags = [marine.tag for marine in self.marines]
            if len(self.enemy_units) > 0:
                enemy = random.choice(self.enemy_units)
                return actions.RAW_FUNCTIONS.Attack_pt("now", tags, (enemy.x, enemy.y))
            attack_xy = (random.randint(1,83), random.randint(1,83))
            return actions.RAW_FUNCTIONS.Attack_pt("now", tags, attack_xy)
        else:
            return actions.RAW_FUNCTIONS.no_op()

    def get_best_scv(self):
        idle_scvs = [scv for scv in self.scvs if scv.order_length == 0]
        scv = None
        if len(idle_scvs) > 0:
            scv = random.choice(idle_scvs)
        else:
            if len(self.scvs) > 0:
                scv = random.choice(self.scvs)

        return scv

    def train_scv(self):
        if self.cc is not None:
            return actions.RAW_FUNCTIONS.Train_SCV_quick("now", self.cc.tag)
        else:
            return actions.RAW_FUNCTIONS.no_op()

    def train_marine(self):
        if len(self.barracks) > 0:
            barracks = random.choice(self.barracks)
            return actions.RAW_FUNCTIONS.Train_Marine_quick("now", barracks.tag)
        else:
            return actions.RAW_FUNCTIONS.no_op()


    def harvest_minerals(self):
        idle_scvs = [scv for scv in self.scvs if scv.order_length == 0]
        if len(idle_scvs) > 0:
            scv = random.choice(idle_scvs)
        else:
            if len(self.scvs) > 0:
                scv = random.choice(self.scvs)
            else:
                return actions.RAW_FUNCTIONS.no_op()
        if scv is not None:
            distances = self.get_distances(self.mineral_patches, (scv.x, scv.y))
            mineral_patch = self.mineral_patches[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Harvest_Gather_SCV_unit("now", scv.tag, mineral_patch.tag)
        else:
            return actions.RAW_FUNCTIONS.no_op()

    def all_scvs_harvest_minerals(self):
        if len(self.scvs) > 0:
            if self.cc is not None:
                distances = self.get_distances(self.mineral_patches, (self.cc.x, self.cc.y))

                patch_tags = [patch.tag for patch in self.mineral_patches][:5]
                tags = [scv.tag for scv in self.scvs]
                return actions.RAW_FUNCTIONS.Harvest_Gather_SCV_unit("now", tags, patch_tags)
            else:
                return actions.RAW_FUNCTIONS.no_op()
        else:
            return actions.RAW_FUNCTIONS.no_op()

    def harvest_gas(self):
        scv = self.get_best_scv()
        if scv is not None:
            if len(self.refineries) > 0:
                gas = random.choice(self.refineries)
                return actions.RAW_FUNCTIONS.Harvest_Gather_SCV_unit("now", scv.tag,gas.tag)
        else:
            return actions.RAW_FUNCTIONS.no_op()


    def build_supply_depot(self):
        if self.cc.any():
            cc = self.cc
        else:
            return actions.RAW_FUNCTIONS.no_op()
        unitx = cc.x
        unity = cc.y
        new_x = unitx + random.randint(-5, 5)
        new_y = unity + random.randint(-5, 5)
        scv = self.get_best_scv()
        if scv is not None:
            return actions.RAW_FUNCTIONS.Build_SupplyDepot_pt("now", scv.tag, (new_x, new_y))
        else:
            return actions.RAW_FUNCTIONS.no_op()

    def build_barracks(self):
        if self.cc.any():
            cc = self.cc
        else:
            return actions.RAW_FUNCTIONS.no_op()
        new_x = cc.x + random.randint(-5, 5)
        new_y = cc.y + random.randint(-5, 5)
        scv = self.get_best_scv()
        if scv is not None:
            return actions.RAW_FUNCTIONS.Build_Barracks_pt("now", scv.tag, (new_x, new_y))
        else:
            return actions.RAW_FUNCTIONS.no_op()

    def build_gas_refinery(self):
        location = None
        scv = self.get_best_scv()
        if scv is not None:
            distances = self.get_distances(self.gas, (scv.x, scv.y))
            geyser = self.gas[np.argmin(distances)]

            if geyser is not None:
                return actions.RAW_FUNCTIONS.Build_Refinery_pt("now", scv.tag, geyser.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def reset(self):
        if self.env is None:
            self.init_env()
        self.marines = []
        self.scvs = []
        raw_obs = self.env.reset()[0]
        return self.get_derived_obs(raw_obs)

    def init_env(self):
        args = {**self.default_settings, **self.kwargs}
        self.env = sc2_env.SC2Env(**args)

    def get_derived_obs(self, raw_obs):
        #print(raw_obs.observation['score_cumulative'][0])
        obs = np.zeros((500, 4), dtype=np.uint8)
        # 1 indicates my own unit, 4 indicates enemy's
        #zerglings = self.get_units_by_type(raw_obs, units.Zerg.Zergling, 4)
        #banelings = self.get_units_by_type(raw_obs, units.Zerg.Baneling, 4)
        cc = self.get_units_by_type(raw_obs, units.Terran.CommandCenter, 1)
        if cc:
            self.cc = cc.pop()
        self.scvs = self.get_units_by_type(raw_obs, units.Terran.SCV, 1)
        self.marines = self.get_units_by_type(raw_obs, units.Terran.Marine, 1)
        self.gas = [unit for unit in raw_obs.observation.raw_units if unit.unit_type == units.Neutral.VespeneGeyser]
        self.refineries = [unit for unit in raw_obs.observation.raw_units if unit.unit_type == units.Terran.Refinery]
        self.barracks = [unit for unit in raw_obs.observation.raw_units if unit.unit_type == units.Terran.Barracks]

        self.enemy_units = self.get_enemy_units(raw_obs)
        self.mineral_patches = [unit for unit in raw_obs.observation.raw_units
            if unit.unit_type in [
               units.Neutral.BattleStationMineralField,
               units.Neutral.BattleStationMineralField750,
               units.Neutral.LabMineralField,
               units.Neutral.LabMineralField750,
               units.Neutral.MineralField,
               units.Neutral.MineralField750,
               units.Neutral.PurifierMineralField,
               units.Neutral.PurifierMineralField750,
               units.Neutral.PurifierRichMineralField,
               units.Neutral.PurifierRichMineralField750,
               units.Neutral.RichMineralField,
               units.Neutral.RichMineralField750
        ]]

        for i, m in enumerate(raw_obs.observation.raw_units):
            try:
                obs[i] = np.array([m.x, m.y, m[2], m.alliance])
            except:
                continue

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
        #print(f"Unit: {[unit for unit in units]}")
        units_xy = [(unit.x, unit.y) for unit in units]
        return np.linalg.norm(np.array(units_xy) - np.array(xy), axis=1)

    def close(self):
        print("CLOSE")
        if self.env is not None:
            self.env.close()
        super().close()

    def render(self, mode='human', close=False):
        pass
