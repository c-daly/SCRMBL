import numpy as np
from pysc2.lib import actions
import random

class SC2UnitManager():
    def __init__(self):
        super().__init__()
        self.movement_rate_per_step = 2
        self.obs = None
        self.enemies = None
        self.enemy_tags = None
        self.units = None
        self.individual_actions = [self.move_up, self.move_down, self.move_left, self.move_right, self.attack_closest_enemy]
        self.ordered_units = []

    def step(self, obs):
        self.obs = obs[0]
        self.units = [unit for unit in self.obs.observation.raw_units if unit.alliance == 1]
        self.enemies = [unit for unit in self.obs.observation.raw_units if unit.alliance == 4]
        self.enemy_tags = [enemy.tag for enemy in self.enemies]
        action_queue = []
        unit = random.choice(self.units)
        action_queue.append(self.move_left(unit))
        return action_queue

    def move_up(self, marine):
        self.ordered_units.append(marine)
        try:
            new_pos = [marine.x, marine.y - self.movement_rate_per_step]
            return actions.RAW_FUNCTIONS.Move_pt("now", marine.tag, new_pos)
        except:
            return actions.RAW_FUNCTIONS.no_op()

    def move_down(self, marine):
        try:
            new_pos = [marine.x, marine.y + self.movement_rate_per_step]
            return actions.RAW_FUNCTIONS.Move_pt("now", marine.tag, new_pos)
        except:
            return actions.RAW_FUNCTIONS.no_op()

    def move_left(self, marine):
        try:
            new_pos = [marine.x - self.movement_rate_per_step, marine.y]
            return actions.RAW_FUNCTIONS.Move_pt("now", marine.tag, new_pos)
        except Exception as err:
            return actions.RAW_FUNCTIONS.no_op()

    def move_right(self, marine):
        try:
            new_pos = [marine.x + self.movement_rate_per_step, marine.y]
            return actions.RAW_FUNCTIONS.Move_pt("now", marine.tag, new_pos)
        except:
            return actions.RAW_FUNCTIONS.no_op()

    def attack(self, marine):
        try:
            enemy = random.choice(self.enemies)
            return actions.RAW_FUNCTIONS.Attack_pt("now", (marine.x, marine.y), (enemy.x, enemy.y))
        except Exception as err:
            return actions.RAW_FUNCTIONS.no_op()

    def attack_closest_enemy(self, marine):
        if len(self.enemies) > 0:
            return actions.RAW_FUNCTIONS.Attack_pt("now", marine.tag, random.choice(self.enemies).tag)
        return actions.RAW_FUNCTIONS.no_op()

    def get_distances(self, units, xy):
        units_xy = [(unit.x, unit.y) for unit in units]
        return np.linalg.norm(np.array(units_xy) - np.array(xy), axis=1)
