from collections import namedtuple
import random
from agents import pysc2_base_agent
from Utils import SC2_utils
from pysc2.lib import actions, units
from actions.SCII.pysc2_actions import action_defs
import numpy as np

SC2_utils = SC2_utils.SC2Utils

NO_STRATEGY = 'NO_STRATEGY'
INFANTRY_RUSH = 'INFANTRY_RUSH'

STRATEGIES = [NO_STRATEGY, INFANTRY_RUSH]

BUILD_RUSHING_ARMY = 'BUILD_RUSHING_ARMY'
SWARM_ENEMY_BASE = 'SWARM_ENEMY_BASE'
HIGH_LEVEL_ACTIONS = [BUILD_RUSHING_ARMY, SWARM_ENEMY_BASE]

GOAL_5_MARINES = 'GOAL_5_MARINES'
GOAL_10_MARINES = 'GOAL_10_MARINES'
GOAL_25_MARINES = 'GOAL_25_MARINES'
GOAL_50_MARINES = 'GOAL_50_MARINES'
GOAL_1_BARRACKS = 'GOAL_1_BARRACKS'
GOAL_2_BARRACKS = 'GOAL_2_BARRACKS'
GOAL_4_BARRACKS = 'GOAL4_BARRACKS'
GOAL_5_DEPOTS = 'GOAL_5_DEPOTS'
GOAL_10_DEPOTS = 'GOAL_10_DEPOTS'
GOAL_25_SCVS = 'GOAL_25_SCVS'
GOAL_35_SCVS = 'GOAL_35_SCVS'


def check_for_x_units_of_type(x, unit_type, obs):
    """
    Vehicle for checking goal satisfaction.
    """
    goal_units = [unit for unit in obs.observation.raw_units if unit.unit_type == unit_type and unit.alliance == 1]
    if len(goal_units) >= x:
        return True
    return False


GoalSatisfactionDefinition = namedtuple("GoalSatisfactionDefinition",
                                          ["method", "x", "unit_type"])

EVALUATE_25_MARINES = GoalSatisfactionDefinition(check_for_x_units_of_type, 35, units.Terran.Marine)
EVALUATE_50_MARINES = GoalSatisfactionDefinition(check_for_x_units_of_type, 50, units.Terran.Marine)
EVALUATE_35_SCVS = GoalSatisfactionDefinition(check_for_x_units_of_type, 35, units.Terran.SCV)
EVALUATE_25_SCVS = GoalSatisfactionDefinition(check_for_x_units_of_type, 25, units.Terran.SCV)
EVALUATE_5_DEPOTS = GoalSatisfactionDefinition(check_for_x_units_of_type, 5, units.Terran.SupplyDepot)
EVALUATE_2_BARRACKS = GoalSatisfactionDefinition(check_for_x_units_of_type, 2, units.Terran.Barracks)

goal_satisfaction = {GOAL_35_SCVS: EVALUATE_35_SCVS,
                     GOAL_25_SCVS: EVALUATE_25_SCVS,
                     GOAL_5_DEPOTS: EVALUATE_5_DEPOTS,
                     GOAL_2_BARRACKS: EVALUATE_2_BARRACKS,
                     GOAL_25_MARINES: EVALUATE_25_MARINES,
                     GOAL_50_MARINES: EVALUATE_50_MARINES}

refinements = {INFANTRY_RUSH: [SWARM_ENEMY_BASE, BUILD_RUSHING_ARMY],
               BUILD_RUSHING_ARMY: [GOAL_50_MARINES, GOAL_2_BARRACKS, GOAL_25_SCVS, GOAL_5_DEPOTS],
               SWARM_ENEMY_BASE: [action_defs.ALL_UNITS_ATTACK]}

MARINE_GOAL_ACTIONS = [action_defs.TRAIN_MARINE,
                       action_defs.TRAIN_SCV,
                       action_defs.BUILD_BARRACKS,
                       action_defs.BUILD_SUPPLY_DEPOT,
                       action_defs.HARVEST_MINERALS]

BARRACKS_GOAL_ACTIONS = [action_defs.TRAIN_SCV,
                         action_defs.BUILD_BARRACKS,
                         action_defs.HARVEST_MINERALS]

SUPPLY_GOAL_ACTIONS = [action_defs.BUILD_SUPPLY_DEPOT,
                       action_defs.HARVEST_MINERALS]

SCV_GOAL_ACTIONS = [action_defs.HARVEST_MINERALS,
                    action_defs.TRAIN_SCV]

actions_by_goal = {GOAL_5_MARINES: MARINE_GOAL_ACTIONS,
                   GOAL_10_MARINES: MARINE_GOAL_ACTIONS,
                   GOAL_25_MARINES: MARINE_GOAL_ACTIONS,
                   GOAL_50_MARINES: MARINE_GOAL_ACTIONS,
                   GOAL_5_DEPOTS: SUPPLY_GOAL_ACTIONS,
                   GOAL_10_DEPOTS: SUPPLY_GOAL_ACTIONS,
                   GOAL_2_BARRACKS: BARRACKS_GOAL_ACTIONS,
                   GOAL_4_BARRACKS: BARRACKS_GOAL_ACTIONS,
                   GOAL_25_SCVS: SCV_GOAL_ACTIONS,
                   GOAL_35_SCVS: SCV_GOAL_ACTIONS}


class HierarchicalScriptedAgent(pysc2_base_agent.Pysc2BaseAgent):
    def __init__(self):
        self.initialize_properties()

    def initialize_properties(self):
        self.strategy = INFANTRY_RUSH
        self.available_actions = refinements[self.strategy]
        self.obs = None
        self.enemy_units = []
        self.marines = []
        self.depots = []
        self.scvs = []
        self.idle_scvs = []
        self.marines = []
        self.cc = None
        self.barracks = []
        self.satisfied_goals = []
        self.high_level_actions = refinements[INFANTRY_RUSH]
        self.current_hla = self.high_level_actions[-1]
        self.goals = refinements[self.current_hla]
        #self.goals = [goal for goal in [refinements[action] for action in self.high_level_actions]][0]
        self.completed_hlas = []
        self.current_goal = self.goals[-1]
        self.current_simple_actions = []
        self.mineral_patches = []
        self.ccx = None
        self.ccy = None
        self.swarming = False
        self.attacking_units = []
        self.last_enemy_location = None

    def reset(self):
        self.initialize_properties()

    def step(self, obs):
        """
        Setting strategy sets high level actions
        Appraise strategy, possibly switch
        Get high level actions for selected strategy
        high level actions drive goals
        Examine unmet goals for current hla/strategy
        Get unperformed actions for current goal
        perform first unperformed action
        """
        self.attacking_units = SC2_utils.get_units_that_can_attack(obs[0])
        self.obs = obs

        if len(self.mineral_patches) == 0:
            self.populate_mineral_patches()

        self.scvs = SC2_utils.get_units_by_type(obs[0], units.Terran.SCV, 1)
        self.enemy_units = SC2_utils.get_enemy_units(obs[0])

        try:
            self.cc = SC2_utils.get_units_by_type(self.obs[0], units.Terran.CommandCenter, 1)[0]
        except:
            self.cc = None

        if self.ccx is None:
            self.ccx = self.cc.x
            self.ccy = self.cc.y

        if len(self.attacking_units) > 85 or self.swarming:
            self.swarming = True
            return self.all_units_attack()
        if self.current_goal:
            if self.check_if_goal_is_satisfied():
                self.satisfied_goals = self.goals.pop()
                #self.current_simple_actions.pop()
                if len(self.goals) > 0:
                    self.current_goal = self.goals[-1]
                else:
                    if len(self.high_level_actions) > 0:
                        self.completed_hlas = self.high_level_actions.pop()
                        if len(self.high_level_actions) > 0:
                            self.current_hla = self.high_level_actions[-1]
                            self.goals = refinements[self.current_hla]
                            self.current_goal = self.goals[-1]
            action = self.get_actions_for_current_goal()
            return self.translate_action(action)
        else:
            self.complete_current_action_and_get_next()
            action = self.get_actions_for_current_goal()
            return self.translate_action(action)
        return actions.RAW_FUNCTIONS.no_op()

    def populate_mineral_patches(self):
        obs = self.obs[0]
        self.mineral_patches = [unit for unit in obs.observation.raw_units if unit.unit_type in [
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

    def translate_action(self, action):
        if action == action_defs.NO_OP:
            return actions.RAW_FUNCTIONS.no_op()
        if action == action_defs.TRAIN_SCV:
            if self.cc is not None:
                return actions.RAW_FUNCTIONS.Train_SCV_quick("now", self.cc.tag)
            else:
                return actions.RAW_FUNCTIONS.no_op()
        else:
            if action == action_defs.TRAIN_MARINE:
                barracks = SC2_utils.get_units_by_type(self.obs[0], units.Terran.Barracks, 1)
                if barracks:
                    barrack = random.choice(barracks)
                    return actions.RAW_FUNCTIONS.Train_Marine_quick("now", barrack.tag)
                else:
                    return actions.RAW_FUNCTIONS.no_op()
        if action == action_defs.BUILD_BARRACKS:
            scv = random.choice(self.scvs)
            unitx = self.ccx
            unity = self.ccy
            new_x = unitx + random.randint(-5, 5)
            new_y = unity + random.randint(-5, 5)
            return actions.RAW_FUNCTIONS.Build_Barracks_pt("now", scv.tag, (new_x, new_y))

        if action == action_defs.BUILD_SUPPLY_DEPOT:
            if len(self.scvs) > 0:
                scv = random.choice(self.scvs)
                unitx = self.ccx
                unity = self.ccy
                new_x = unitx + random.randint(-5, 5)
                new_y = unity + random.randint(-5, 5)
                return actions.RAW_FUNCTIONS.Build_SupplyDepot_pt("now", scv.tag, (new_x, new_y))
        else:
            return actions.RAW_FUNCTIONS.no_op()
        if action == action_defs.ALL_UNITS_ATTACK:
            return self.all_units_attack()
        if action == action_defs.HARVEST_MINERALS:
            scv = self.get_best_scv()
            if scv is not None:
                distances = SC2_utils.get_distances(self.mineral_patches, (scv.x, scv.y))
                patch = self.mineral_patches[np.argmin(distances)]
                return actions.RAW_FUNCTIONS.Harvest_Gather_SCV_unit("now", scv, patch.tag)
            else:
                return actions.RAW_FUNCTIONS.no_op()

    def check_if_goal_is_satisfied(self):
        if self.current_goal in action_defs.ALL_ACTIONS:
            return False
        goal_eval = goal_satisfaction[self.current_goal]
        unit_type = goal_eval.unit_type
        x = goal_eval.x
        eval_method = goal_eval.method
        satisfied = eval_method(x, unit_type, self.obs[0])

        return satisfied

    def all_units_attack(self):
        """
        Return all mobile units (SCVs and marines)
        """
        tags = [unit.tag for unit in self.attacking_units]
        if not self.enemy_units:
            if self.last_enemy_location is not None:
                position = self.last_enemy_location
            else:
                position = SC2_utils.get_random_position()
            if len(tags) > 0:
                return actions.RAW_FUNCTIONS.Move_pt("now", tags, position)
            else:
                return actions.RAW_FUNCTIONS.no_op()
        enemy = random.choice(self.enemy_units)
        self.last_enemy_location = enemy.x, enemy.y
        attack_xy = (enemy.x, enemy.y)
        if len(tags) > 0:
            return actions.RAW_FUNCTIONS.Attack_pt("now", tags, attack_xy)
        else:
            return actions.RAW_FUNCTIONS.no_op()

    def complete_current_action_and_get_next(self):
        self.completed_hlas.append(self.high_level_actions.pop())
        if self.high_level_actions:
            self.current_hla = self.high_level_actions[-1]
            self.goals = refinements[self.current_hla]
        if not self.goals:
            # Not sure what to do here.  Don't want to
            # overload actions.  Should be able to
            # just grab the next action and populate
            # the goals
            return
        self.current_goal = self.goals[-1]

    def get_actions_for_current_goal(self):
        """
        The actions available are currently
        only limited by goal.  This method
        will use the observation data to
        choose between available actions under
        the goal
        """
        if isinstance(self.current_goal, list):
            goal = self.current_goal[0]
        else:
            goal = self.current_goal

        # Some goals are just actions
        if goal in action_defs.ALL_ACTIONS:
            return goal
        try:
            action = random.choice(actions_by_goal[goal])
            #action = actions_by_goal[goal][-1]
        except Exception as ex:
            print(f"{ex}")
        return action

    def strategy_is_failing(self):
        """
        Currently no work but this method
        will be used to judge the efficacy
        of the current strategy. A strategy
        might fail because it's a poor
        strategy for the moment, but also
        because it's implemented poorly in
        terms of its goals/action definitions.
        """
        return False

    def translate_action_to_pysc2(self, action):
        return actions.RAW_FUNCTIONS.no_op()

    # def get_units_by_type(self, obs, unit_type, player_relative=0):
    #     """
    #     NONE = 0
    #     SELF = 1
    #     ALLY = 2
    #     NEUTRAL = 3
    #     ENEMY = 4
    #     """
    #     return [unit for unit in obs.observation.raw_units
    #             if unit.unit_type == unit_type
    #             and unit.alliance == player_relative]

    def get_best_scv(self):
        best_scv = None
        if len(self.scvs) == 0:
            return None
        if self.idle_scvs is None:
            self.idle_scvs = [scv for scv in self.scvs if scv.order_length == 0]
            best_scv = random.choice(self.idle_scvs)
        else:
            best_scv = random.choice(self.scvs)
        return best_scv

    # def get_random_position(self):
    #     return random.randint(25, 35), random.randint(25, 35)
    #
    # def get_enemy_units(self, obs):
    #     return [unit for unit in obs.observation.raw_units if unit.alliance == 4]
    #
    # def get_units_that_can_attack(self, obs):
    #     attacking_units = [unit for unit in obs.observation.raw_units if unit.alliance == 1
    #                        and (unit.unit_type == units.Terran.Marine or unit.unit_type == units.Terran.SCV)]
    #     return attacking_units
    #
    # def get_distances(self, units, xy):
    #     # print(f"Unit: {[unit for unit in units]}")
    #     units_xy = [(unit.x, unit.y) for unit in units]
    #     return np.linalg.norm(np.array(units_xy) - np.array(xy), axis=1)

    def reset(self):
        return
