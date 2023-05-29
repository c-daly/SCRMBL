from s2clientprotocol import sc2api_pb2 as sc_pb
from s2clientprotocol import raw_pb2 as raw_pb
from s2clientprotocol import common_pb2 as common_pb
class SC2ProcessManager(object):
    def __init__(self, websocket, scenario, step_multiplier=1):
        self.websocket = websocket
        self.scenario = scenario
        self.step_multiplier = step_multiplier

    def create_game(self):
        response = None
        create_game = sc_pb.RequestCreateGame(
            realtime=False,
            # local_map=sc_pb.LocalMap(map_path="melee/Simple64.SC2Map")
            local_map=sc_pb.LocalMap(map_path="mini_games/" + self.scenario.map_name),
            # player_setup=[
            #    sc_pb.PlayerSetup(type=sc_pb.Participant)  # Human
            #    #sc_pb.PlayerSetup(type=sc_pb.Computer)  # Human
            # ]
        )

        create_game.player_setup.add(type=sc_pb.Participant, race=common_pb.Terran)
        create_game.player_setup.add(type=sc_pb.Computer, race=common_pb.Zerg, difficulty=sc_pb.Easy)
        request = sc_pb.Request(create_game=create_game)
        try:
            self.websocket.send(request.SerializeToString())
            response_data = self.websocket.recv()
            response = sc_pb.Response.FromString(response_data)
        except Exception as e:
            print(f"Create game failure: {e}")

        if len(response.error) > 0:
            print(f"Error: {response.error}")
            return

        # Join the game
        join_game = sc_pb.RequestJoinGame(
            race=common_pb.Terran,
            options=sc_pb.InterfaceOptions(
                raw=True,
                score=True,
                feature_layer=sc_pb.SpatialCameraSetup(
                    width=64,

                ),
                render=sc_pb.SpatialCameraSetup(width=64,),
            )
        )
        request = sc_pb.Request(join_game=join_game)
        try:
            self.websocket.send(request.SerializeToString())
            response_data = self.websocket.recv()
            response = sc_pb.Response.FromString(response_data)

            if len(response.error) > 0:
                print(f"Error: {response.error}")
                return
        except Exception as e:
            print(f"Join Error {e}")

    def get_obs(self):
        request = sc_pb.Request(observation=sc_pb.RequestObservation())
        self.websocket.send(request.SerializeToString())
        response_data = self.websocket.recv()
        response = sc_pb.Response.FromString(response_data)
        return self.scenario.get_derived_obs_from_raw(response.observation)


    def step(self):
        # Step the game forward by a single step
        request_step = sc_pb.RequestStep(count=self.step_multiplier)
        request = sc_pb.Request(step=request_step)
        self.websocket.send(request.SerializeToString())
        response_data = self.websocket.recv()

        response = sc_pb.Response.FromString(response_data)
        return response