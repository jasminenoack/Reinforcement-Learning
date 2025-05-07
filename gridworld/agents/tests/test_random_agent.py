import random

from gridworld.utils import RIGHT, UP, DOWN, LEFT
from ..random_agent import RandomAgent


class TestAct:
    def test_should_return_random_action(self):
        rng = random.Random(42)
        agent = RandomAgent(rng=rng)
        assert agent.act((0, 0)) == UP
        assert agent.act((0, 0)) == UP
        assert agent.act((0, 0)) == LEFT
        assert agent.act((0, 0)) == DOWN
        assert agent.act((0, 0)) == DOWN
        assert agent.act((0, 0)) == DOWN
        assert agent.act((0, 0)) == UP
        assert agent.act((0, 0)) == UP
        assert agent.act((0, 0)) == RIGHT
