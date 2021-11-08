from envs.SCRMBLEnv import SCRMBLEnv
from agents.Hierarchical_scripted_agent import HierarchicalScriptedAgent
from agents.simple_agent import SimpleAgent
from absl import flags
from pysc2.env import environment

FLAGS = flags.FLAGS
FLAGS([''])

def main():
    #agent = HierarchicalScriptedAgent()
    agent = SimpleAgent()
    env = SCRMBLEnv()
    try:
        while True:
            # New Episode
            agent.reset()
            obs = env.reset()
            while True:
                action = agent.step(obs)
                obs = env.step([action])
                if type(obs[0]) == environment.StepType.LAST:
                    print(f"Episode score: {obs[0].observation.score_cumulative[0]}")
                    break
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()