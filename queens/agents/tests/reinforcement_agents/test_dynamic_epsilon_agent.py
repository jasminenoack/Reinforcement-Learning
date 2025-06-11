import random

import pytest
import numpy as np

from queens.agents.reinforcement_agents import DynamicEpsilonAgent
from queens.dtos import StepResult, RunnerReturn
from queens.utils import build_board_array

board = build_board_array([])


class TestEpsilonFinalIncrease:
    def test_returns_scaled_value(self):
        agent = DynamicEpsilonAgent(epsilon=0.2, epsilon_increase=1.5)
        assert agent.epsilon_final_increase() == pytest.approx(0.2 * 1.5)


class TestObserveResult:
    def test_updates_on_success(self):
        rng = random.Random(1)
        agent = DynamicEpsilonAgent(
            rng=rng, epsilon=0.5, epsilon_decay=0.5, epsilon_min=0.1
        )
        agent.observe_result(
            RunnerReturn(
                trajectory=[StepResult(action=(0, 0))],
                solved=True,
                board=np.array([[0]]),
                moves=1,
                score=100,
            )
        )
        assert agent.epsilon == pytest.approx(max(0.1, 0.5 * 0.5))
        assert agent.failure_count == 0
        assert agent.last_failure_minus_one == []

    def test_updates_on_failure(self):
        rng = random.Random(2)
        agent = DynamicEpsilonAgent(
            rng=rng,
            epsilon=0.2,
            epsilon_increase=1.5,
            epsilon_max=0.5,
        )
        agent.observe_result(
            RunnerReturn(
                trajectory=[StepResult(action=(0, 0))],
                solved=False,
                board=np.array([[0]]),
                moves=1,
                score=-1,
            )
        )
        expected = min(0.5, 0.2 * 1.5)
        assert agent.epsilon == pytest.approx(expected)
        assert agent.failure_count == 1
