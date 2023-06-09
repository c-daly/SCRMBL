import gym.spaces

from envs.BaseEnv import BaseEnv
from s2clientprotocol import sc2api_pb2 as sc_pb
from s2clientprotocol import raw_pb2 as raw_pb
from managers.SC2ProcessManager import SC2ProcessManager
import numpy as np
from scenarios.DefeatZerglingsAndBanelingsScenario import DefeatZerglingsAndBanelingsScenario
from envs.SC2.SC2Actions import Actions

class SC2SyncEnvExtended(BaseEnv):
    def __init__(self, websocket, scenario=None, step_multiplier=16, **kwargs):
        super().__init__()
        self.scenario = scenario
        self.step_multiplier = step_multiplier
        self.default_move_speed = 2
        if scenario is None:
            self.scenario = DefeatZerglingsAndBanelingsScenario()
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
        #self.action_space = self.scenario.action_space
        self.action_space = gym.spaces.MultiDiscrete([15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15])
        self.scenario.action_space = self.action_space
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
            #self.get_marines()
            self.take_action(action)
            # Step the game forward by a single step
            response = self.sc2_manager.step()

            if response.status != 3:
                done = True
                self.sc2_manager.create_game()
                #self.reset()

            self.reward = self.scenario.raw_obs.observation.score.score
            #step = self.scenario.raw_obs.observation.game_loop/self.step_multiplier
            #if step == 0:
            #    step = 1
            #self.reward = self.scenario.raw_obs.observation.score.score * 1/step

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
            cmd_func = Actions.attack_enemy
            actions_pb = []
            self.get_obs()
            self.get_marines()
            #for i, marine in enumerate(self.marines):
            if np.isscalar(action):
                action = [action]
            for i, marine in enumerate(self.marines):
                cmd = action[i]
                if cmd == 1:
                    cmd_func = Actions.move_left
                elif cmd == 2:
                    cmd_func = Actions.move_right
                elif cmd == 3:
                    cmd_func = Actions.move_down
                elif cmd == 4:
                    cmd_func = Actions.move_up
                elif cmd > 4:
                    cmd_func = Actions.attack_enemy

                #actions_pb.append(raw_pb.ActionRaw(unit_command=self.random_attack(i, action[i])))

                # cmd == 4 is a noop
                if 0 < cmd < 5:
                    if len(self.marines) > i:
                        actions_pb.append(raw_pb.ActionRaw(unit_command=cmd_func(self.marines[i], action[i], self.default_move_speed)))
                else:
                    if len(self.enemies) > i:
                        actions_pb.append(raw_pb.ActionRaw(unit_command=cmd_func(self.marines[i], self.enemies[i])))


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