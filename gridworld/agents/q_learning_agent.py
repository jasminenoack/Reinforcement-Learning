from collections import defaultdict
from random import Random
from gridworld.agents.generic_agent import Agent
from gridworld.utils import SIMPLE_ACTIONS


class QLearningAgent(Agent):
    def __init__(self, *, rng: Random | None = None):
        self.actions = SIMPLE_ACTIONS
        self.q_table = {action: 0 for action in self.actions}
        # exploration rate
        self.epsilon = 0.1
        # learning rate (how fast Q-values update)
        self.alpha = 0.1
        # discount factor (importance of future reward)
        self.gamma = 0.9
        self.rng = rng or Random()
