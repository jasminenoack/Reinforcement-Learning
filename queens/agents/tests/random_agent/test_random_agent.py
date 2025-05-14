from queens.agents.random_agent import RandomAgent
from queens.dtos import BoardState, Observation
import random

from queens.utils import build_board_array


class TestAct:
    def test_random_agent(self):
        rng = random.Random(42)
        agent = RandomAgent(rng=rng)
        board = build_board_array([])
        assert agent.act(Observation(board_state=BoardState(board=board))) == (1, 0)
        assert agent.act(Observation(board_state=BoardState(board=board))) == (4, 3)
        assert agent.act(Observation(board_state=BoardState(board=board))) == (3, 2)
        assert agent.act(Observation(board_state=BoardState(board=board))) == (1, 1)
        assert agent.act(Observation(board_state=BoardState(board=board))) == (6, 0)
        assert agent.act(Observation(board_state=BoardState(board=board))) == (0, 1)
        assert agent.act(Observation(board_state=BoardState(board=board))) == (3, 3)
