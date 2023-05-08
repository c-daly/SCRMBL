import asyncio
import random

from s2clientprotocol import sc2api_pb2 as sc_pb
from s2clientprotocol import common_pb2 as common_pb
from websockets import connect


async def main():
    async with connect("ws://127.0.0.1:5000/sc2api") as websocket:
        # Create a game
        create_game = sc_pb.RequestCreateGame(
            realtime=True,
            local_map=sc_pb.LocalMap(map_path="mini_games/DefeatZerglingsAndBanelings.SC2Map")
        )

        create_game.player_setup.add(type=sc_pb.Participant)
        join_game = sc_pb.RequestJoinGame(
            race=common_pb.Terran,
            options=sc_pb.InterfaceOptions(raw=True)
        )

        request = sc_pb.Request(create_game=create_game)
        await websocket.send(request.SerializeToString())
        response_data = await websocket.recv()
        response = sc_pb.Response.FromString(response_data)

        #if response.status != 0:
        #    print(f"Error: {response.error}")
        #    return

        # Join the game
        join_game = sc_pb.RequestJoinGame(
            race=common_pb.Terran,
            options=sc_pb.InterfaceOptions(raw=True)
        )
        request = sc_pb.Request(join_game=join_game)
        await websocket.send(request.SerializeToString())
        response_data = await websocket.recv()
        response = sc_pb.Response.FromString(response_data)

        #if response.status != 0:
        #    print(f"Error: {response.error}")
        #    return

        # Main observation-action loop
        for _ in range(10):
            # Get observation
            request = sc_pb.Request(observation=sc_pb.RequestObservation())
            await websocket.send(request.SerializeToString())
            response_data = await websocket.recv()
            response = sc_pb.Response.FromString(response_data)
            observation = response.observation

            # Perform a random action (move a random unit)
            actions_pb = []
            for unit in observation.observation.raw_data.units:
                if unit.alliance == common_pb.Self:
                #    target = common_pb.Point2D(
                #        x=random.uniform(0, 200),
                #        y=random.uniform(0, 200)
                #    )
                #    action = sc_pb.ActionRawUnitCommand(
                #        ability_id=3,  # Move
                #        unit_tags=[unit.tag],
                #        target_world_space_pos=target
                #    )
                #    actions_pb.append(sc_pb.ActionRaw(unit_command=action))

            request_action = sc_pb.RequestAction(actions=actions_pb)
            request = sc_pb.Request(action=request_action)
            await websocket.send(request.SerializeToString())
            response_data = await websocket.recv()
            response = sc_pb.Response.FromString(response_data)

            #if response.status != 0:
            #    print(f"Error: {response.error}")
            #    return

            # Sleep for a moment before the next iteration
            await asyncio.sleep(1)

        # Leave the game
        request_leave_game = sc_pb.RequestLeaveGame()
        request = sc_pb.Request(leave_game=request_leave_game)
        await websocket.send(request.SerializeToString())
        response_data = await websocket.recv()
        response = sc_pb.Response.FromString(response_data)

        if response.status != sc_pb.Response.Status.success:
            print(f"Error: {response.error}")
            return


def run_main():
    asyncio.run(main())


if __name__ == "__main__":
    run_main()
