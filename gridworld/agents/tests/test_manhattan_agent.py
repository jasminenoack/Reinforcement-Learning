from gridworld.agents.manhattan_agent import ManhattanAgent
from gridworld.utils import DOWN, RIGHT


class TestAct:
    def test_goes_right_and_down(self):
        agent = ManhattanAgent()
        agent.reset()
        assert agent.act((0, 0)) in [DOWN, RIGHT]
        assert agent.act((1, 0)) in [DOWN, RIGHT]
        assert agent.act((2, 0)) in [DOWN, RIGHT]
        assert agent.act((3, 0)) in [DOWN, RIGHT]
        assert agent.act((4, 0)) in [DOWN, RIGHT]
        assert agent.act((4, 1)) in [DOWN, RIGHT]
        assert agent.act((4, 2)) in [DOWN, RIGHT]
        assert agent.act((4, 3)) in [DOWN, RIGHT]
