from gridworld.agents.manhattan_agent import ManhattanAgent
from gridworld.utils import DOWN, LEFT, RIGHT, UP, Step


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

    def test_does_a_previously_unseen_action(self):
        agent = ManhattanAgent()
        agent._action_cache = {
            (1, 1): set([DOWN, RIGHT, LEFT]),
        }
        assert agent.act((1, 1)) == UP

    def test_goes_back_if_has_no_valid_options(self):
        agent = ManhattanAgent()
        agent.reset()
        agent._action_cache = {
            (1, 1): set([DOWN, RIGHT, LEFT, UP]),
        }
        agent._backtrack = [UP]
        assert agent.act((1, 1)) == UP


class TestReset:
    def test_sets_action_cache(self):
        agent = ManhattanAgent()
        agent.reset()
        assert agent._action_cache == {}

    def test_sets_history(self):
        agent = ManhattanAgent()
        agent.reset()
        assert agent._backtrack == []


class TestObserve:
    def test_adds_reverse_action_to_backtrack(self):
        agent = ManhattanAgent()
        agent.reset()
        agent.observe(
            Step(
                start=(2, 0),
                action=DOWN,
                new_state=(2, 1),
                reward=0,
                done=False,
            )
        )
        assert agent._backtrack == [UP]

    def test_adds_previous_action_to_history(self):
        agent = ManhattanAgent()
        agent.reset()
        agent.observe(
            Step(
                start=(2, 0),
                action=DOWN,
                new_state=(2, 1),
                reward=0,
                done=False,
            )
        )
        assert agent._action_cache == {(2, 0): set([DOWN]), (2, 1): set()}

    def test_do_not_add_a_backtrack_to_backtrack(self):
        agent = ManhattanAgent()
        agent.reset()
        agent._backtrack = []
        agent._action_cache[(2, 1)] = set([UP])
        agent.observe(
            Step(
                start=(2, 0),
                action=DOWN,
                new_state=(2, 1),
                reward=0,
                done=False,
            )
        )
        assert agent._backtrack == []
