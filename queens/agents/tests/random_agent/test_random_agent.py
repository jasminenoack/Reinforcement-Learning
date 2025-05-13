from queens.agents.random_agent import RandomAgent
from queens.dtos import Observation
import random


class TestAct:
    def test_random_agent(self):
        rng = random.Random(42)
        agent = RandomAgent(rng=rng)
        assert agent.act(Observation()) == (1, 0)
        assert agent.act(Observation()) == (4, 3)
        assert agent.act(Observation()) == (3, 2)
        assert agent.act(Observation()) == (1, 1)
        assert agent.act(Observation()) == (6, 0)
        assert agent.act(Observation()) == (0, 1)
        assert agent.act(Observation()) == (3, 3)
