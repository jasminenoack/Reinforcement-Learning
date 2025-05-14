import numpy as np
from queens.agents.reinforcement_agents import SimpleRandomReinforcementAgent
from queens.dtos import BoardState, Observation, StepResult, RunnerReturn
import random

from queens.utils import build_board_array

board = build_board_array([])


class TestAct:
    def test_act_follows_path_most_of_times(self):
        rng = random.Random(42)
        agent = SimpleRandomReinforcementAgent(rng=rng)
        agent.following_path = [(0, 0), (1, 1), (2, 2)]
        agent.current = [(0, 0), (1, 1)]
        assert agent.act(
            observation=Observation(board_state=BoardState(board=board))
        ) == (2, 2)

    def test_act_follows_random_path_without_options(self):
        rng = random.Random(1)
        agent = SimpleRandomReinforcementAgent(rng=rng)
        agent.following_path = []
        agent.current = []
        assert agent.act(
            observation=Observation(board_state=BoardState(board=board))
        ) == (1, 4)

    def test_will_sometimes_follow_random_path(self):
        rng = random.Random(1)
        agent = SimpleRandomReinforcementAgent(rng=rng)
        agent.following_path = [(0, 0), (1, 1), (2, 2)]
        agent.current = [(0, 0), (1, 1)]
        assert agent.act(
            observation=Observation(board_state=BoardState(board=board))
        ) == (2, 2)
        agent.current = [(0, 0), (1, 1)]
        assert agent.act(
            observation=Observation(board_state=BoardState(board=board))
        ) == (2, 2)
        agent.current = [(0, 0), (1, 1)]
        assert agent.act(
            observation=Observation(board_state=BoardState(board=board))
        ) == (2, 2)
        agent.current = [(0, 0), (1, 1)]
        assert agent.act(
            observation=Observation(board_state=BoardState(board=board))
        ) == (2, 2)
        agent.current = [(0, 0), (1, 1)]
        assert agent.act(
            observation=Observation(board_state=BoardState(board=board))
        ) == (2, 2)
        agent.current = [(0, 0), (1, 1)]
        assert agent.act(
            observation=Observation(board_state=BoardState(board=board))
        ) == (2, 2)
        agent.current = [(0, 0), (1, 1)]
        assert agent.act(
            observation=Observation(board_state=BoardState(board=board))
        ) == (2, 2)
        agent.current = [(0, 0), (1, 1)]
        assert agent.act(
            observation=Observation(board_state=BoardState(board=board))
        ) == (2, 2)
        agent.current = [(0, 0), (1, 1)]
        # epsilon
        assert agent.act(
            observation=Observation(board_state=BoardState(board=board))
        ) == (0, 6)
        agent.current = [(0, 0), (1, 1)]
        assert agent.act(
            observation=Observation(board_state=BoardState(board=board))
        ) == (2, 2)
        agent.current = [(0, 0), (1, 1)]
        assert agent.act(
            observation=Observation(board_state=BoardState(board=board))
        ) == (2, 2)
        agent.current = [(0, 0), (1, 1)]
        #  epsilon
        assert agent.act(
            observation=Observation(board_state=BoardState(board=board))
        ) == (7, 4)
        agent.current = [(0, 0), (1, 1)]
        assert agent.act(
            observation=Observation(board_state=BoardState(board=board))
        ) == (2, 2)


class TestObserveStep:
    pass


class TestObserveResults:
    pass


class TestReset:
    def reset(self):
        rng = random.Random(1)
        agent = SimpleRandomReinforcementAgent(rng=rng)
        agent.best_options = {  # type: ignore
            1: [[(0, 0), (1, 1)], [(2, 2), (3, 3)]],
            2: [[(4, 4), (5, 5)], [(6, 6), (7, 7)]],
        }
        agent.reset()
        assert agent.current == []
        assert agent.following_path == [(0, 0), (1, 1)]


class TestObserveResult:
    def test_saves_up_to_row_collisions(self):
        rng = random.Random(1)
        agent = SimpleRandomReinforcementAgent(rng=rng)
        agent.observe_result(
            RunnerReturn(
                trajectory=[
                    StepResult(
                        action=(0, 0),
                    ),
                    StepResult(
                        action=(1, 7),
                    ),
                    StepResult(
                        action=(0, 2),
                    ),
                ],
                solved=False,
                board=np.array([[0]]),
                moves=8,
                score=-100,
            )
        )
        assert agent.best_options == {
            -98: [[(0, 0), (1, 7)]],
        }

    def test_saves_up_to_column_collisions(self):
        rng = random.Random(1)
        agent = SimpleRandomReinforcementAgent(rng=rng)
        agent.current = [(0, 0), (1, 7), (2, 7)]
        agent.observe_result(
            RunnerReturn(
                trajectory=[
                    StepResult(
                        action=(0, 0),
                    ),
                    StepResult(
                        action=(1, 7),
                    ),
                    StepResult(
                        action=(2, 7),
                    ),
                ],
                solved=False,
                board=np.array([[0]]),
                moves=8,
                score=-100,
            )
        )
        assert agent.best_options == {
            -98: [[(0, 0), (1, 7)]],
        }

    def test_saves_up_to_diagonal_collisions(self):
        rng = random.Random(1)
        agent = SimpleRandomReinforcementAgent(rng=rng)
        agent.observe_result(
            RunnerReturn(
                trajectory=[
                    StepResult(
                        action=(0, 0),
                    ),
                    StepResult(
                        action=(1, 7),
                    ),
                    StepResult(
                        action=(1, 1),
                    ),
                ],
                solved=False,
                board=np.array([[0]]),
                moves=8,
                score=-100,
            )
        )
        assert agent.best_options == {
            -98: [[(0, 0), (1, 7)]],
        }

    def test_saves_up_to_reverse_diagonal_collisions(self):
        rng = random.Random(1)
        agent = SimpleRandomReinforcementAgent(rng=rng)
        agent.observe_result(
            RunnerReturn(
                trajectory=[
                    StepResult(
                        action=(0, 0),
                    ),
                    StepResult(
                        action=(1, 7),
                    ),
                    StepResult(
                        action=(2, 6),
                    ),
                ],
                solved=False,
                board=np.array([[0]]),
                moves=8,
                score=-100,
            )
        )
        assert agent.best_options == {
            -98: [[(0, 0), (1, 7)]],
        }
