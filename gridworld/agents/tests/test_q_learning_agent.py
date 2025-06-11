import random
from gridworld.agents.q_learning_agent import QLearningAgent
from gridworld.utils import LEFT, RIGHT, UP, DOWN, Step


class TestAct:
    def test_returns_random_action_when_epsilon(self, mocker):
        rng = mocker.Mock()
        rng.random.return_value = 0.05
        agent = QLearningAgent(rng=rng)
        agent.q_table = {(0, 0): {UP: 0.0, DOWN: 1.0, LEFT: 1.0, RIGHT: 0.0}}
        agent.act((0, 0))
        rng.choice.assert_called_once_with(agent.actions)

    def test_returns_best_action_when_not_epsilon(self, mocker):
        rng = mocker.Mock()
        rng.random.return_value = 0.2
        agent = QLearningAgent(rng=rng)
        agent.q_table = {(0, 0): {UP: 0.0, DOWN: 1.0, LEFT: 1.0, RIGHT: 0.0}}
        agent.act((0, 0))
        rng.choice.assert_called_once_with([DOWN, LEFT])

    def test_returns_an_action_with_max_q_value(self, mocker):
        rng = random.Random(42)
        agent = QLearningAgent(rng=rng)
        agent.q_table = {(0, 0): {UP: 1.0, DOWN: 0.0, LEFT: 0.0, RIGHT: 0.0}}
        action = agent.act((0, 0))
        assert action == UP


class TestReset:
    def test_do_not_reset_q_table(self):
        agent = QLearningAgent()
        old_table = {(0, 0): {UP: 1.0, DOWN: 0.0, LEFT: 0.0, RIGHT: 0.0}}
        agent.q_table = old_table
        agent.reset()
        assert isinstance(agent.q_table, dict)
        assert agent.q_table == old_table


class TestObserve:
    def test_initial_q_table_update(self):
        agent = QLearningAgent()
        agent.q_table[(0, 0)] = {UP: 0.0, DOWN: 0.0, LEFT: 0.0, RIGHT: 0.0}
        step = Step(
            start=(0, 0),
            action=UP,
            reward=-10,
            new_state=(0, 0),
            done=False,
        )
        agent.observe(step)
        assert agent.q_table == {
            (0, 0): {UP: -1.0, DOWN: 0.0, LEFT: 0.0, RIGHT: 0.0},
        }

    def test_handles_more_information_if_it_knows_something_already(self):
        agent = QLearningAgent()
        agent.q_table[(0, 0)] = {UP: -6.0, DOWN: 0.0, LEFT: 0.0, RIGHT: 0.0}
        step = Step(
            start=(0, 0),
            action=UP,
            reward=-10,
            new_state=(0, 1),
            done=False,
        )
        agent.observe(step)
        assert agent.q_table == {
            (0, 0): {UP: -6.4, DOWN: 0.0, LEFT: 0.0, RIGHT: 0.0},
            (0, 1): {UP: 0.0, DOWN: 0.0, LEFT: 0.0, RIGHT: 0.0},
        }

    def test_updates_if_already_at_score(self):
        agent = QLearningAgent()
        agent.q_table[(0, 0)] = {UP: -10.0, DOWN: 0.0, LEFT: 0.0, RIGHT: 0.0}
        step = Step(
            start=(0, 0),
            action=UP,
            reward=-10,
            new_state=(0, 0),
            done=True,
        )
        agent.observe(step)
        assert agent.q_table == {
            (0, 0): {UP: -10.0, DOWN: 0.0, LEFT: 0.0, RIGHT: 0.0},
        }

    def test_updates_for_next_location_score(self):
        agent = QLearningAgent()
        agent.q_table[(1, 0)] = {UP: 0.0, DOWN: 0.0, LEFT: 10.0, RIGHT: 0.0}
        step = Step(
            start=(0, 0),
            action=DOWN,
            reward=-1,
            new_state=(1, 0),
            done=False,
        )
        agent.observe(step)
        assert agent.q_table == {
            (0, 0): {UP: 0.0, DOWN: 0.8, LEFT: 0.0, RIGHT: 0.0},
            (1, 0): {UP: 0.0, DOWN: 0.0, LEFT: 10.0, RIGHT: 0.0},
        }

    def test_next_best_choice_is_negative(self):
        agent = QLearningAgent()
        agent.q_table[(1, 0)] = {UP: -10.0, DOWN: -1.0, LEFT: -10.0, RIGHT: -1.0}
        step = Step(
            start=(0, 0),
            action=DOWN,
            reward=-1,
            new_state=(1, 0),
            done=False,
        )
        agent.observe(step)
        assert agent.q_table == {
            (0, 0): {UP: 0.0, DOWN: -0.19, LEFT: 0.0, RIGHT: 0.0},
            (1, 0): {UP: -10.0, DOWN: -1.0, LEFT: -10.0, RIGHT: -1.0},
        }
