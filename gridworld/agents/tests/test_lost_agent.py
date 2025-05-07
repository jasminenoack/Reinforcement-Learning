import random

from gridworld.utils import RIGHT, UP, DOWN, LEFT, Step
from ..lost_agent import LostAgent


class TestAct:
    def test_should_return_random_action(self):
        rng = random.Random(42)
        agent = LostAgent(rng=rng)
        assert agent.act((0, 0)) == UP
        assert agent.act((0, 0)) == UP
        assert agent.act((0, 0)) == LEFT
        assert agent.act((0, 0)) == DOWN
        assert agent.act((0, 0)) == DOWN
        assert agent.act((0, 0)) == DOWN
        assert agent.act((0, 0)) == UP
        assert agent.act((0, 0)) == UP
        assert agent.act((0, 0)) == RIGHT

    def test_should_avoid_used_actions(self):
        rng = random.Random(42)
        agent = LostAgent(rng=rng)
        agent.reset()
        agent.observe(
            Step(
                start=(0, 0),
                action=UP,
                new_state=(0, 1),
                reward=0,
                done=False,
            )
        )
        assert agent.act((0, 0)) == RIGHT


class TestReset:
    def test_should_reset_previous_attempts(self):
        rng = random.Random(42)
        agent = LostAgent(rng=rng)
        agent.reset()
        assert agent.previous_attempts == {}


class TestObserve:
    def test_should_add_action_to_previous_attempts(self):
        rng = random.Random(42)
        agent = LostAgent(rng=rng)
        agent.reset()
        agent.observe(
            Step(
                start=(0, 0),
                action=UP,
                new_state=(0, 1),
                reward=0,
                done=False,
            )
        )
        assert agent.previous_attempts == {
            (0, 0): {UP: 1, DOWN: 0, LEFT: 0, RIGHT: 0},
            (0, 1): {UP: 0, DOWN: 1, LEFT: 0, RIGHT: 0},
        }
        agent.observe(
            Step(
                start=(0, 0),
                action=UP,
                new_state=(0, 1),
                reward=0,
                done=False,
            )
        )
        assert agent.previous_attempts[(0, 0)][UP] == 2
        assert agent.previous_attempts[(0, 1)][DOWN] == 2
        agent.observe(
            Step(
                start=(0, 0),
                action=DOWN,
                new_state=(0, 1),
                reward=0,
                done=False,
            )
        )
        assert agent.previous_attempts[(0, 0)][DOWN] == 1
        assert agent.previous_attempts[(0, 1)][UP] == 1

    def test_handle_no_reverse_action(self):
        rng = random.Random(42)
        agent = LostAgent(rng=rng)
        agent.reset()
        step = Step(
            start=(0, 0),
            action=UP,
            new_state=(0, 0),
            reward=0,
            done=False,
        )
        agent.observe(step)
        assert agent.previous_attempts == {
            (0, 0): {UP: 1, DOWN: 0, LEFT: 0, RIGHT: 0},
        }
