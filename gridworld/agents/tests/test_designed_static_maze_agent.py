from gridworld.agents.designed_agent import DesignedStaticMazeAgent
from gridworld.utils import DOWN, RIGHT


class TestReset:
    def test_should_reset_path(self):
        agent = DesignedStaticMazeAgent()
        agent.reset()
        assert agent._path == [RIGHT] * 4 + [DOWN] * 4


class TestAct:
    def test_goes_right_and_down(self):
        agent = DesignedStaticMazeAgent()
        agent.reset()
        assert agent.act((0, 0)) == RIGHT
        assert agent.act((0, 1)) == RIGHT
        assert agent.act((0, 2)) == RIGHT
        assert agent.act((0, 3)) == RIGHT
        assert agent.act((0, 4)) == DOWN
        assert agent.act((1, 4)) == DOWN
        assert agent.act((2, 4)) == DOWN
        assert agent.act((3, 4)) == DOWN
