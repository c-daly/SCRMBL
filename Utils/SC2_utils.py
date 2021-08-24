from pysc2.lib import units, actions
import numpy as np

class SC2Utils:

    def get_units_by_type(obs, unit_type, player_relative=0):
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

    def get_random_position():
        return random.randint(25, 35), random.randint(25, 35)

    def get_enemy_units(obs):
        return [unit for unit in obs.observation.raw_units if unit.alliance == 4]

    def get_units_that_can_attack(obs):
        attacking_units = [unit for unit in obs.observation.raw_units if unit.alliance == 1
                           and (unit.unit_type == units.Terran.Marine or unit.unit_type == units.Terran.SCV)]
        return attacking_units

    def get_distances(units, xy):
        units_xy = [(unit.x, unit.y) for unit in units]
        return np.linalg.norm(np.array(units_xy) - np.array(xy), axis=1)
