import asyncio
import random

from s2clientprotocol import sc2api_pb2 as sc_pb
from s2clientprotocol import common_pb2 as common_pb
from websockets import connect

# Custom action functions
# ... (unchanged)

async def main():
    async with connect("ws://127.0.0.1:5000/sc2api") as websocket:
        # Create a game
        create_game = sc_pb.RequestCreateGame(
            realtime=False,
            local_map=sc_pb.LocalMap(map_path="FindAndDefeatZerglings.SC2Map"),
            player_setup=[
                sc_pb.PlayerSetup(type=sc_pb.Participant),  # Human
                sc_pb.PlayerSetup(type=sc_pb.Computer, race=common_pb.Random)  # AI
            ]
        )
        request = sc_pb.Request(create_game=create_game)
        await websocket.send(request.SerializeToString())
        response_data = await websocket.recv()
        response = sc_pb.Response.FromString(response_data)

        if len(response.error) > 0:
            print(f"Error(s): {response.error}")
            return

        # Join the game
        # ... (unchanged)

        # Main observation-action loop
        # ... (unchanged)

# ... (run_main function and main check unchanged)
