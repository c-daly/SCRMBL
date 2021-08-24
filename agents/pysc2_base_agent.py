from pysc2.agents.base_agent import BaseAgent

NO_OP = 'NO_OP'

ACTIONS = [NO_OP]
class Pysc2BaseAgent(BaseAgent):
    """
    Base agent for all Pysc2 agents (gym-based or not)
    """
    def __init__(self):
        super(BaseAgent, self).__init__()

    def step(self, obs):
        return NO_OP

