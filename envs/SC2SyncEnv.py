#from gym.spaces import Discrete
from gym.spaces import Box, Discrete, MultiDiscrete, Tuple

from envs.BaseEnv import BaseEnv
from gym import spaces
from s2clientprotocol import sc2api_pb2 as sc_pb
from s2clientprotocol import raw_pb2 as raw_pb
from s2clientprotocol import common_pb2 as common_pb
from managers.SC2ProcessManager import SC2ProcessManager
import numpy as np
from scenarios.SC2MiniMapScenario import SC2MiniMapScenario, MoveToBeaconScenario, DefeatRoachesScenario

import random

class SC2SyncEnv(BaseEnv):
    def __init__(self, websocket, **kwargs):
        super().__init__()
        self.scenario = SC2MiniMapScenario()
        self.last_kill_value = 0
        self.sc2_manager = SC2ProcessManager(websocket, self.scenario)
        self.derived_obs = []
        self.enemies_killed_last_step = 0
        self.marines = []
        self.enemies = []
        self.enemies_killed = 0
        self.websocket = websocket
        self.observation = None
        self.last_reward = 0
        self.reward = 0
        self.done = False
        self.info = None
        self.action_space = self.scenario.action_space #MultiDiscrete([4, 4, 4, 4, 4, 4, 4, 4, 4])
        #self.map_high = 128
        #self.map = np.zeros((self.map_high, self.map_high), dtype=int)
        #self.observation_space = Box(
        #    low=-1,
        #    high=10,
        #    shape=(self.map_high,self.map_high),
        #    dtype=np.int
        #)
        self.observation_space = self.scenario.observation_space
        #self.action_space = Discrete(9)
        #self.observation_space = Discrete(9)
        self.sc2_manager.create_game()

    #def print_pixelmap(self):
    #    self.get_marines()
    #    self.map = np.zeros((self.map_high, self.map_high), dtype=int)
    #    for i, marine in enumerate(self.marines):
    #        x = int(marine.pos.x)
    #        y = int(marine.pos.y)
    #        self.map[x][y] = i

    #    for i, enemy in enumerate(self.enemies):
    #        x = int(enemy.pos.x)
    #        y = int(enemy.pos.y)
    #        self.map[x][y] = -1
        #for x in range(self.map_high):
        #    for y in range(self.map_high):
        #        print(f"({x},{y}")
    #    return(self.map)
    #def restart_game(self):
    #    self.sc2_manager.create_game()
    #    self.reset()

    #def create_game(self):
    #    response = None
    #    create_game = sc_pb.RequestCreateGame(
    #        realtime=False,
    #        # local_map=sc_pb.LocalMap(map_path="melee/Simple64.SC2Map")
    #        local_map=sc_pb.LocalMap(map_path="mini_games/DefeatZerglingsAndBanelings.SC2Map"),
    #        # player_setup=[
    #        #    sc_pb.PlayerSetup(type=sc_pb.Participant)  # Human
    #        #    #sc_pb.PlayerSetup(type=sc_pb.Computer)  # Human
    #        # ]
    #    )

    #    create_game.player_setup.add(type=sc_pb.Participant, race=common_pb.Terran)
    #    create_game.player_setup.add(type=sc_pb.Computer, race=common_pb.Zerg, difficulty=sc_pb.Easy)
    #    request = sc_pb.Request(create_game=create_game)
    #    try:
    #        self.websocket.send(request.SerializeToString())
    #        response_data = self.websocket.recv()
    #        response = sc_pb.Response.FromString(response_data)
    #    except Exception as e:
    #        print(f"Create game failure: {e}")

    #    if len(response.error) > 0:
    #        print(f"Error: {response.error}")
    #        return

        # Join the game
    #    join_game = sc_pb.RequestJoinGame(
    #        race=common_pb.Terran,
    #        options=sc_pb.InterfaceOptions(
    #            raw=True,

    #        )
    #    )
    #    request = sc_pb.Request(join_game=join_game)
    #    try:
    #        self.websocket.send(request.SerializeToString())
    #        response_data = self.websocket.recv()
    #        response = sc_pb.Response.FromString(response_data)

    #        if len(response.error) > 0:
    #            print(f"Error: {response.error}")
    #            return
    #    except Exception as e:
    #        print(f"Join Error {e}")

    def reset(self):
        self.obs = None
        #self.create_game()
        self.derived_obs = []
        self.last_reward = 0
        self.reward = 0
        self.enemies_killed = 0
        self.enemies_killed_last_step = 0
        self.done = False
        self.info = {}
        self.marines = []
        self.enemies = []
        return self.get_obs()


    def get_obs(self):
        try:
        #    request = sc_pb.Request(observation=sc_pb.RequestObservation())
        #    try:
        #        self.websocket.send(request.SerializeToString())
        #        response_data = self.websocket.recv()
        #        response = sc_pb.Response.FromString(response_data)
        #    except Exception as e:
        #        print(f"Error {e}")
            self.obs = self.sc2_manager.get_obs()
            self.raw_obs = self.obs
            #observation = self.scneario.get_derived_obs_from_raw(self.obs.observation.raw_data.units)
            #self.raw_obs =  observation
            derived_obs = self.scenario.map #[]
            x_s = []
            y_s = []
            #for i in range(19):
            #for unit in response.observation.observation.raw_data.units:
            #    if len(observation) > i:
            #        unit = observation[i]
            #        #new_obs = ((unit.pos.x + 0.1) * (unit.pos.y + 0.1))
                    #new_obs = ((unit.pos.x + .001)/8) * ((unit.pos.y + .001)/8)
            #        new_obs = [unit.pos.x, unit.pos.y]
            #        x_s.append(unit.pos.x)
            #        y_s.append(unit.pos.y)
                    #derived_obs.append(np.log((unit.pos.x + 0.1) * (unit.pos.y + .1)))
            #        derived_obs.append(new_obs)
            #    else:
            #        x_s.append(-1)
            #        y_s.append(-1)
            #        derived_obs.append([-1,-1])
            #self.derived_obs = derived_obs
            #self.derived_obs = self.print_pixelmap()
            #self.derived_obs = [x_s, y_s]
        except Exception as e:
            print(f"get_obs error: {e}")

        return self.obs

    def get_marines(self):
        #self.marines = [unit for unit in self.raw_obs if unit.alliance == 1]
        self.marines = self.scenario.marines
        return self.marines

    def step(self, action):
        done = False
        #if len(self.obs.player_result) > 0:
        #    self.sc2_manager.create_game()
            #raise Exception("Game ended")

            # get feature map
            #self.reset()
        #self.enemies = [unit for unit in self.raw_obs if unit.alliance != 1]
        try:
            self.get_marines()
            self.take_action(action)
            # Step the game forward by a single step

        #    request_step = sc_pb.RequestStep(count=2)
        #    request = sc_pb.Request(step=request_step)
        #    self.websocket.send(request.SerializeToString())
        #    response_data = self.websocket.recv()
            response = self.sc2_manager.step()
            #response = sc_pb.Response.FromString(response_data)
            if response.status != 3:
                if response.status == 5:
                    self.sc2_manager.create_game()
                    done = True
                    #self.reset()
            #self.reward = len(self.marines) - len(self.enemies) + enemies_killed
            #self.reward = (self.obs.observation.score.score_details.killed_value_units - self.last_kill_value)/(response.step.simulation_loop + 1) #- response.step.simulation_loop
            self.reward = self.scenario.raw_obs.observation.score.score # - self.last_reward
            #self.last_reward = self.reward
            #print(self.reward)


            #self.reward = np.log(max((len(self.marines) + enemies_killed)/max(len(self.enemies), 1), 1))
            #print(f"Reward: {self.reward}")

        except Exception as e:
            print(f"Step error: {e}")
            #self.restart_game()
        return self.obs, self.reward, done, self.info
    def render(self):
        pass

    def close(self):
        pass

    def random_move(self, unit, pos):
        unit_tag = self.marines[unit].tag
        x = pos % 64
        y = pos / 64
        return raw_pb.ActionRawUnitCommand(
            ability_id=3674,  # Move
            unit_tags=[unit_tag],
            target_world_space_pos=common_pb.Point2D(x=x, y=y)
    )

    def random_attack(self, unit):
        try:
            unit_tag = self.marines[unit].tag
            x = np.random.uniform(0,64)
            y = np.random.uniform(0,64)
            return raw_pb.ActionRawUnitCommand(
                ability_id=23,  # Move
                unit_tags=[unit_tag],
                target_world_space_pos=common_pb.Point2D(x=x, y=y)
            )
        except Exception as e:
            print(f"attack action error {e}")


    def move_left(self, unit, pos):
        try:
            marine = self.marines[unit]
            unit_tag = self.marines[unit].tag
            x = marine.pos.x - 2
            y = marine.pos.y
            return raw_pb.ActionRawUnitCommand(
                ability_id=23,  # Move
                unit_tags=[unit_tag],
                target_world_space_pos=common_pb.Point2D(x=x, y=y)
            )
        except Exception as e:
            print(f"move left error {e}")

    def move_right(self, unit, pos):
        marine = self.marines[unit]
        unit_tag = self.marines[unit].tag
        x = marine.pos.x + 2
        y = marine.pos.y
        return raw_pb.ActionRawUnitCommand(
            ability_id=23,  # Move
            unit_tags=[unit_tag],
            target_world_space_pos=common_pb.Point2D(x=x, y=y)
        )
    def move_up(self, unit, pos):
        marine = self.marines[unit]
        unit_tag = self.marines[unit].tag
        x = marine.pos.x
        y = marine.pos.y - 2
        return raw_pb.ActionRawUnitCommand(
            ability_id=23,  # Move
            unit_tags=[unit_tag],
            target_world_space_pos=common_pb.Point2D(x=x, y=y)
        )

    def move_down(self, unit, pos):
        marine = self.marines[unit]
        unit_tag = self.marines[unit].tag
        x = marine.pos.x
        y = marine.pos.y + 2
        return raw_pb.ActionRawUnitCommand(
            ability_id=23,  # Move
            unit_tags=[unit_tag],
            target_world_space_pos=common_pb.Point2D(x=x, y=y)
        )

    def attack_closest_enemy(unit, enemy_units):
        closest_enemy = None
        min_distance_sq = float("inf")

        for enemy in enemy_units:
            dx = unit.pos.x - enemy.pos.x
            dy = unit.pos.y - enemy.pos.y
            distance_sq = dx * dx + dy * dy

            if distance_sq < min_distance_sq:
                min_distance_sq = distance_sq
                closest_enemy = enemy

        if closest_enemy is not None:
            return raw_pb.ActionRawUnitCommand(
                ability_id=23,  # Attack
                unit_tags=[unit.tag],
                target_unit_tag=closest_enemy.tag
            )
        else:
            return None

    def take_action(self, action):
        try:
            actions_pb = []
            self.get_obs()
            self.get_marines()
            #for i, marine in enumerate(self.marines):
            if np.isscalar(action):
                action = [action]
            for i, marine in enumerate(self.marines):
                if i < 9:
                    cmd = action[i]
                    if cmd == 0:
                        cmd_func = self.move_left
                    elif cmd == 1:
                        cmd_func = self.move_right
                    elif cmd == 2:
                        cmd_func = self.move_down
                    elif cmd == 3:
                        cmd_func = self.move_up
                    else:
                        raise Exception("Invalid action")
                    #actions_pb.append(raw_pb.ActionRaw(unit_command=self.random_attack(i, action[i])))
                    actions_pb.append(raw_pb.ActionRaw(unit_command=cmd_func(i, action[i])))

                else:
                    actions_pb.append(raw_pb.ActionRaw(unit_command=self.random_attack(i)))
                    break

            request_action = sc_pb.RequestAction(actions=[sc_pb.Action(action_raw=a) for a in actions_pb])
            request = sc_pb.Request(action=request_action)
            self.websocket.send(request.SerializeToString())
            #try:
            response_data = self.websocket.recv()
            #except Exception as e:
            response = sc_pb.Response.FromString(response_data)

            if len(response.error) > 0:
                for result in response.action.result:
                    print(f"Action Result: {result}")
        except Exception as e:
             print(f"take action error {e}")