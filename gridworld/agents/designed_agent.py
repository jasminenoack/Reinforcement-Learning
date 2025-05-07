import random
from gridworld.agents.generic_agent import Agent
from gridworld.utils import DOWN, RIGHT


class DesignedStaticMazeAgent(Agent):
    def reset(self, **kwargs):
        self._path = [RIGHT] * 4 + [DOWN] * 4

    def act(self, state):
        if self._path:
            return self._path.pop(0)
        return random.sample([RIGHT, DOWN])  # filler if somehow we keep going

    def observe(self, *args, **kwargs):
        pass
