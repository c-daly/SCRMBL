import asyncio
import random

import websockets.exceptions
from s2clientprotocol import sc2api_pb2 as sc_pb
from s2clientprotocol import raw_pb2 as raw_pb
from s2clientprotocol import common_pb2 as common_pb
from websockets import connect

def move_up(unit, distance=1):
    return raw_pb.ActionRawUnitCommand(
        ability_id=2,  # Move
        unit_tags=[unit.tag],
        target_world_space_pos=common_pb.Point2D(x=unit.pos.x, y=min(unit.pos.y + distance, 63))
    )

def move_down(unit, distance=1):
    return raw_pb.ActionRawUnitCommand(
        ability_id=2,  # Move
        unit_tags=[unit.tag],
        target_world_space_pos=common_pb.Point2D(x=unit.pos.x, y=max(unit.pos.y - distance, 0))
    )

def move_left(unit, distance=1):
    return raw_pb.ActionRawUnitCommand(
        ability_id=23,  # Move
        unit_tags=[unit.tag],
        target_world_space_pos=common_pb.Point2D(x=max(unit.pos.x - distance, 0), y=unit.pos.y)
    )

def move_right(unit, distance=1):
    return raw_pb.ActionRawUnitCommand(
        ability_id=23,  # Move
        unit_tags=[unit.tag],
        target_world_space_pos=common_pb.Point2D(x=min(unit.pos.x + distance, 63), y=unit.pos.y)
    )

def random_move(unit):
    return raw_pb.ActionRawUnitCommand(
        ability_id=23,  # Move
        unit_tags=[unit.tag],
        target_world_space_pos=common_pb.Point2D(x=random.uniform(0, 1039), y=random.uniform(0,1039))
    )

async def main():
    async with connect("ws://127.0.0.1:5000/sc2api") as websocket:
        # Create a game
        create_game = sc_pb.RequestCreateGame(
            realtime=True,
            local_map=sc_pb.LocalMap(map_path="melee/Simple64.SC2Map")
            #local_map=sc_pb.LocalMap(map_path="mini_games/DefeatZerglingsAndBanelings.SC2Map"),
            #player_setup=[
            #    sc_pb.PlayerSetup(type=sc_pb.Participant)  # Human
            #    #sc_pb.PlayerSetup(type=sc_pb.Computer)  # Human
            #]
        )

        create_game.player_setup.add(type=sc_pb.Participant, race=common_pb.Terran)
        create_game.player_setup.add(type=sc_pb.Computer, race=common_pb.Zerg, difficulty=sc_pb.Easy)
        request = sc_pb.Request(create_game=create_game)
        await websocket.send(request.SerializeToString())
        response_data = await websocket.recv()
        response = sc_pb.Response.FromString(response_data)

        if len(response.error) > 0:
            print(f"Error: {response.error}")
            return

        # Join the game
        join_game = sc_pb.RequestJoinGame(
            race=common_pb.Terran,
            options=sc_pb.InterfaceOptions(raw=True)
        )
        request = sc_pb.Request(join_game=join_game)
        await websocket.send(request.SerializeToString())
        response_data = await websocket.recv()
        response = sc_pb.Response.FromString(response_data)

        if len(response.error) > 0:
            print(f"Error: {response.error}")
            return

        # Main observation-action loop
        #for _ in range(1000):
        while True:
            try:
                # Get observation
                request = sc_pb.Request(observation=sc_pb.RequestObservation())
                await websocket.send(request.SerializeToString())
                response_data = await websocket.recv()
                response = sc_pb.Response.FromString(response_data)
                observation = response.observation

                # Perform a random action
                actions_pb = []
                for unit in observation.observation.raw_data.units:
                    if unit.alliance == 1:
                        action_func = random.choice([move_up, move_down, move_left, move_right])
                        action = random_move(unit)
                        actions_pb.append(raw_pb.ActionRaw(unit_command=action))

                request_action = sc_pb.RequestAction(actions=[sc_pb.Action(action_raw=a) for a in actions_pb])
                request = sc_pb.Request(action=request_action)
                await websocket.send(request.SerializeToString())
                response_data = await websocket.recv()
                response = sc_pb.Response.FromString(response_data)

                if len(response.error) > 0:
                    for result in response.action.result:
                        print(f"Action Result: {result}")

                # Step the game forward by a single step
                #request_step = sc_pb.RequestStep(count=1)
                #request = sc_pb.Request(step=request_step)
                #await websocket.send(request.SerializeToString())
                #response_data = await websocket.recv()
                #response = sc_pb.Response.FromString(response_data)

                # Sleep for a moment before the next iteration
                #await asyncio.sleep(1)
            except websockets.exceptions.ConnectionClosed as e:
                print(f"error: {e}")
def run_main():
    asyncio.run(main())

if __name__ == "__main__":
    run_main()