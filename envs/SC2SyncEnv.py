#from gym.spaces import Discrete
from gym.spaces import Box, Discrete, MultiDiscrete

from envs.BaseEnv import BaseEnv
from gym import spaces
from s2clientprotocol import sc2api_pb2 as sc_pb
from s2clientprotocol import raw_pb2 as raw_pb
from s2clientprotocol import common_pb2 as common_pb
import numpy as np

import random

class SC2SyncEnv(BaseEnv):
    def __init__(self, websocket, **kwargs):
        super().__init__()
        self.enemies_killed_last_step = 0
        self.marines = []
        self.enemies = []
        self.enemies_killed = 0
        self.websocket = websocket
        self.observation = None
        self.reward = -10
        self.done = False
        self.info = None
        self.action_space = MultiDiscrete([4096, 4096, 4096, 4096, 4096, 4096,4096, 4096, 4096])
        self.observation_space = Box(
            low=0,
            high=256,
            shape=(19,),
            dtype=np.float32
        )
        #self.action_space = Discrete(9)
        #self.observation_space = Discrete(9)
        self.create_game()

    def restart_game(self):
        self.create_game()
        self.reset()

    def create_game(self):
        create_game = sc_pb.RequestCreateGame(
            realtime=False,
            # local_map=sc_pb.LocalMap(map_path="melee/Simple64.SC2Map")
            local_map=sc_pb.LocalMap(map_path="mini_games/DefeatZerglingsAndBanelings.SC2Map"),
            # player_setup=[
            #    sc_pb.PlayerSetup(type=sc_pb.Participant)  # Human
            #    #sc_pb.PlayerSetup(type=sc_pb.Computer)  # Human
            # ]
        )

        create_game.player_setup.add(type=sc_pb.Participant, race=common_pb.Terran)
        create_game.player_setup.add(type=sc_pb.Computer, race=common_pb.Zerg, difficulty=sc_pb.Easy)
        request = sc_pb.Request(create_game=create_game)
        self.websocket.send(request.SerializeToString())
        response_data = self.websocket.recv()
        response = sc_pb.Response.FromString(response_data)

        if len(response.error) > 0:
            print(f"Error: {response.error}")
            return

        # Join the game
        join_game = sc_pb.RequestJoinGame(
            race=common_pb.Terran,
            options=sc_pb.InterfaceOptions(
                raw=True,
            )
        )
        request = sc_pb.Request(join_game=join_game)
        self.websocket.send(request.SerializeToString())
        response_data = self.websocket.recv()
        response = sc_pb.Response.FromString(response_data)

        if len(response.error) > 0:
            print(f"Error: {response.error}")
            return

    def reset(self):
        #self.create_game()
        self.reward = 0
        self.enemies_killed = 0
        self.enemies_killed_last_step = 0
        self.done = False
        self.info = {}
        self.marines = []
        self.enemies = []
        return self.get_obs()


    def get_obs(self):
        request = sc_pb.Request(observation=sc_pb.RequestObservation())
        self.websocket.send(request.SerializeToString())
        response_data = self.websocket.recv()
        try:
            response = sc_pb.Response.FromString(response_data)
        except Exception as e:
            print(f"Error {e}")

        observation = response.observation.observation.raw_data.units
        self.obs = response.observation
        self.raw_obs = response.observation.observation.raw_data.units
        derived_obs = []
        for i in range(19):
        #for unit in response.observation.observation.raw_data.units:
            if len(response.observation.observation.raw_data.units) > i:
                unit = response.observation.observation.raw_data.units[i]
                derived_obs.append((unit.pos.x + 0) * (unit.pos.y + .0000000001))
            else:
                derived_obs.append(0)
        self.derived_obs = derived_obs
        return self.derived_obs

    def get_marines(self):
        self.marines = [unit for unit in self.raw_obs if unit.alliance == 1]
        return self.marines

    def step(self, action):
        if len(self.obs.player_result) > 0:
            self.restart_game()
            #raise Exception("Game ended")
        # get feature map
        #self.reset()
        self.get_marines()
        self.enemies = [unit for unit in self.raw_obs if unit.alliance != 1]
        self.take_action(action)
        enemies_killed = max(9 - len(self.enemies), 0)
        # Step the game forward by a single step
        request_step = sc_pb.RequestStep(count=1)
        request = sc_pb.Request(step=request_step)
        try:
            self.websocket.send(request.SerializeToString())
        except Exception as e:
            print(f"Error {e}")
        try:
            response_data = self.websocket.recv()
        except Exception as e:
            print(f"Error {e}")
        response = sc_pb.Response.FromString(response_data)
        if response.status != 3:
            if response.status == 5:
                #self.create_game()
                self.done = True
                #self.reset()
        self.enemies_killed += enemies_killed
        #self.reward = len(self.marines) - len(self.enemies) + enemies_killed
        self.reward = np.log(max((len(self.marines) + enemies_killed)/max(len(self.enemies), 1), 0))
        #print(f"Reward: {self.reward}")
        return self.derived_obs, self.reward, self.done, self.info
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

    def random_attack(self, unit, pos):
        unit_tag = self.marines[unit].tag
        x = pos % 64
        y = pos / 64
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
        self.get_obs()
        self.get_marines()
        actions_pb = []
        #for i, marine in enumerate(self.marines):
        for i, marine in enumerate(self.marines):
            actions_pb.append(raw_pb.ActionRaw(unit_command=self.random_attack(i, action[i])))


        #for unit in self.observation.observation.raw_data.units:
        #    if unit.alliance == 1:
        #        action_func = random.choice([move_up, move_down, move_left, move_right])
        #        action = random_move(unit, action)
        #        actions_pb.append(raw_pb.ActionRaw(unit_command=action))

        request_action = sc_pb.RequestAction(actions=[sc_pb.Action(action_raw=a) for a in actions_pb])
        request = sc_pb.Request(action=request_action)
        self.websocket.send(request.SerializeToString())
        try:
            response_data = self.websocket.recv()
        except Exception as e:
            print(f"Error {e}")
        response = sc_pb.Response.FromString(response_data)


        corrective_actions = []
        for i, result in enumerate(response.action.result):
            if result > 1:
                corrective_actions.append(raw_pb.ActionRaw(unit_command=self.random_move(i, action[i])))

        if len(corrective_actions) > 0:
            self.get_marines()
            request_action = sc_pb.RequestAction(actions=[sc_pb.Action(action_raw=a) for a in corrective_actions])
            request = sc_pb.Request(action=request_action)
            self.websocket.send(request.SerializeToString())
            try:
                response_data = self.websocket.recv()
            except Exception as e:
                print(f"Error {e}")
            response = sc_pb.Response.FromString(response_data)

        #if 1 not in response.action.result:
        #    raise Exception()
        if len(response.error) > 0:
            for result in response.action.result:
                print(f"Action Result: {result}")
