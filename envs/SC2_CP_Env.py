from s2clientprotocol import sc2api_pb2 as sc_pb
from s2clientprotocol import raw_pb2 as raw_pb
from s2clientprotocol import common_pb2 as common_pb
import gym
from gym.spaces import spaces

import websockets.exceptions
from s2clientprotocol import sc2api_pb2 as sc_pb
from s2clientprotocol import raw_pb2 as raw_pb
from s2clientprotocol import common_pb2 as common_pb
import websocket
from contextlib import closing

class SC2_CP_Env(gym.Env):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_space = spaces.box(
            low=0,
            high=5,
            shape=(11,)
        )
        self.observation_space = spaces.box(
            low=0,
            high=64,
            shape=(19, 3)
        )

    def move_up(self, unit, distance=1):
        return raw_pb.ActionRawUnitCommand(
            ability_id=23,  # Move
            unit_tags=[unit.tag],
            target_world_space_pos=common_pb.Point2D(x=unit.pos.x, y=min(unit.pos.y + distance, 63))
        )

    def move_down(self, unit, distance=1):
        return raw_pb.ActionRawUnitCommand(
            ability_id=23,  # Move
            unit_tags=[unit.tag],
            target_world_space_pos=common_pb.Point2D(x=unit.pos.x, y=max(unit.pos.y - distance, 0))
        )

    def move_left(self, unit, distance=1):
        return raw_pb.ActionRawUnitCommand(
            ability_id=23,  # Move
            unit_tags=[unit.tag],
            target_world_space_pos=common_pb.Point2D(x=max(unit.pos.x - distance, 0), y=unit.pos.y)
        )

    def move_right(self, unit, distance=1):
        return raw_pb.ActionRawUnitCommand(
            ability_id=23,  # Move
            unit_tags=[unit.tag],
            target_world_space_pos=common_pb.Point2D(x=min(unit.pos.x + distance, 63), y=unit.pos.y)
        )

    def random_move(self, unit):
        return raw_pb.ActionRawUnitCommand(
            ability_id=23,  # Move
            unit_tags=[unit.tag],
            target_world_space_pos=common_pb.Point2D(x=random.uniform(0, 1039), y=random.uniform(0, 1039))
        )

    async def send(self, message):
        await self.websocket.send(message)
    def step(self, action):
        actions_pb = []


        for unit in observation.observation.raw_data.units:
            if unit.alliance == 1:
                #action_func = random.choice([self.move_up, self.move_down, self.move_left, self.move_right])
                action = self.random_move(unit)
                actions_pb.append(raw_pb.ActionRaw(unit_command=action))

        request_action = sc_pb.RequestAction(actions=[sc_pb.Action(action_raw=a) for a in actions_pb])
        request = sc_pb.Request(action=request_action)
        websocket.send(request.SerializeToString())
        response_data = await websocket.recv()
        response = sc_pb.Response.FromString(response_data)

        if len(response.error) > 0:
            for result in response.action.result:
                print(f"Action Result: {result}")

        request = sc_pb.Request(observation=sc_pb.RequestObservation())
        response_data = await websocket.recv()
        response = sc_pb.Response.FromString(response_data)
        observation = response.observation
        return observation
    def reset(self):
        pass

    def render(self):
        pass

    def close(self):
        pass

