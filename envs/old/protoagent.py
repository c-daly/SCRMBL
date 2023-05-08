import asyncio
import random

from s2clientprotocol import sc2api_pb2 as sc_pb
from s2clientprotocol import common_pb2 as common_pb
from s2clientprotocol import raw_pb2 as raw_pb
from websockets import connect


async def main():
    async with connect("ws://127.0.0.1:5000/sc2api") as websocket:
        # Create a game
        create_game = sc_pb.RequestCreateGame(
            realtime=True,
            local_map=sc_pb.LocalMap(map_path="mini_games/DefeatRoaches.SC2Map")
        )

        create_game.player_setup.add(type=sc_pb.Participant)
        create_game.player_setup.add(type=sc_pb.Computer)
        request = sc_pb.Request(create_game=create_game)
        await websocket.send(request.SerializeToString())
        response_data = await websocket.recv()
        response = sc_pb.Response.FromString(response_data)

        # Join the game
        join_game = sc_pb.RequestJoinGame(
            race=common_pb.Terran,
            options=sc_pb.InterfaceOptions(raw=True)
        )
        request = sc_pb.Request(join_game=join_game)
        await websocket.send(request.SerializeToString())
        response_data = await websocket.recv()
        response = sc_pb.Response.FromString(response_data)


        # Main observation-action loop
        for _ in range(10):
            # Get observation
            request = sc_pb.Request(observation=sc_pb.RequestObservation())
            test = request.SerializeToString()
            #await websocket.send(request.SerializeToString())
            await websocket.send(test)
            response_data = await websocket.recv()
            response = sc_pb.Response.FromString(response_data)
            observation = response.observation

            # Perform a random action
            actions_pb = []
            for unit in observation.observation.raw_data.units:
                if unit.alliance == 1:
                    target = common_pb.Point2D(
                        x=random.uniform(0, 255),
                        y=random.uniform(0, 255)
                    )
                    action = raw_pb.ActionRawUnitCommand(
                        ability_id=3,  # Move
                        unit_tags=[unit.tag],
                        target_world_space_pos=target
                    )
                    actions_pb.append(raw_pb.ActionRaw(unit_command=action))
            #unit = observation.observation.raw_data.units[0]
            #action = raw_pb.ActionRawUnitCommand(ability_id=3, unit_tags=[unit.tag], target_world_space_pos=target)


            #compiled_actions = (sc_pb.Action(action_raw=a) for a in actions_pb)
            request_action = sc_pb.RequestAction(actions=[sc_pb.Action(action_raw=a) for a in actions_pb])
            request = sc_pb.Request(action=request_action)
            action2 = await websocket.send(request.SerializeToString())
            response_data = await websocket.recv()
            response = sc_pb.Response.FromString(response_data)

            # Step the game forward by a single step
            #request_step = sc_pb.RequestStep(count=16)
            #request = sc_pb.Request(step=request_step)
            #await websocket.send(request.SerializeToString())
            #response_data = await websocket.recv()
            #response = sc_pb.Response.FromString(response_data)

            # Sleep for a moment before the next iteration
            await asyncio.sleep(1)


def run_main():
    asyncio.run(main())


if __name__ == "__main__":
    run_main()
