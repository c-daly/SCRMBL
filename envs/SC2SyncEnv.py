from envs.BaseEnv import BaseEnv
from s2clientprotocol import sc2api_pb2 as sc_pb
from s2clientprotocol import raw_pb2 as raw_pb
from managers.SC2ProcessManager import SC2ProcessManager
import numpy as np
from scenarios.SC2MiniMapScenario import SC2MiniMapScenario, MoveToBeaconScenario, DefeatRoachesScenario
from envs.SC2.SC2Actions import Actions

class SC2SyncEnv(BaseEnv):
    def __init__(self, websocket, scenario=None, step_multiplier=16, **kwargs):
        super().__init__()
        self.scenario = scenario
        self.default_move_speed = 4
        if scenario is None:
            self.scenario = MoveToBeaconScenario()
        self.last_kill_value = 0
        self.sc2_manager = SC2ProcessManager(websocket, self.scenario, step_multiplier=step_multiplier)
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
        self.action_space = self.scenario.action_space
        self.observation_space = self.scenario.observation_space
        self.sc2_manager.create_game()

    def reset(self):
        self.obs = None
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
            self.obs = self.sc2_manager.get_obs()
            self.raw_obs = self.obs
        except Exception as e:
            print(f"get_obs error: {e}")

        return self.obs

    def get_marines(self):
        self.marines = self.scenario.marines
        return self.marines

    def step(self, action):
        done = False
        try:
            self.get_marines()
            self.take_action(action)
            # Step the game forward by a single step
            response = self.sc2_manager.step()

            if response.status != 3:
                done = True
                self.sc2_manager.create_game()
                #self.reset()

            self.reward = self.scenario.raw_obs.observation.score.score # - self.last_reward

        except Exception as e:
            print(f"Step error: {e}")
            #self.restart_game()
        return self.obs, self.reward, done, self.info
    def render(self):
        pass

    def close(self):
        pass

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
                        cmd_func = Actions.move_left
                    elif cmd == 1:
                        cmd_func = Actions.move_right
                    elif cmd == 2:
                        cmd_func = Actions.move_down
                    elif cmd == 3:
                        cmd_func = Actions.move_up
                    else:
                        raise Exception("Invalid action")
                    #actions_pb.append(raw_pb.ActionRaw(unit_command=self.random_attack(i, action[i])))
                    actions_pb.append(raw_pb.ActionRaw(unit_command=cmd_func(self.marines[i], action[i])))

                else:
                    actions_pb.append(raw_pb.ActionRaw(unit_command=Actions.random_attack(self.marines[i], 64, 64)))
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