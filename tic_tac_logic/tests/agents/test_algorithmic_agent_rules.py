import pytest
from tic_tac_logic.agents.algorithmic_agent import AlgorithmicAgent
from tic_tac_logic.constants import X, O, E


class TestRuleAvoidTriplesRow:
    @pytest.mark.parametrize(
        "grid, expected_move, expected_grid",
        [
            (
                [[X, X, E, E], [E, E, E, E], [E, E, E, E], [E, E, E, E]],
                (0, 2),
                [[X, X, O, E], [E, E, E, E], [E, E, E, E], [E, E, E, E]],
            ),
            (
                [[E, X, X, E], [E, E, E, E], [E, E, E, E], [E, E, E, E]],
                (0, 0),
                [[O, X, X, E], [E, E, E, E], [E, E, E, E], [E, E, E, E]],
            ),
            (
                [[X, E, X, E], [E, E, E, E], [E, E, E, E], [E, E, E, E]],
                (0, 1),
                [[X, O, X, E], [E, E, E, E], [E, E, E, E], [E, E, E, E]],
            ),
            (
                [[X, O, E, E], [E, E, E, E], [E, E, E, E], [E, E, E, E]],
                None,
                [[X, O, E, E], [E, E, E, E], [E, E, E, E], [E, E, E, E]],
            ),
        ],
    )
    def test_rule(self, grid, expected_move, expected_grid):
        agent = AlgorithmicAgent(rows=len(grid), columns=len(grid[0]))
        move = agent._rule_avoid_triples_row(grid)
        assert move == expected_move
        assert grid == expected_grid


class TestRuleAvoidTriplesColumn:
    @pytest.mark.parametrize(
        "grid, expected_move, expected_grid",
        [
            (
                [[X, E, E, E], [X, E, E, E], [E, E, E, E], [E, E, E, E]],
                (2, 0),
                [[X, E, E, E], [X, E, E, E], [O, E, E, E], [E, E, E, E]],
            ),
            (
                [[E, E, E, E], [X, E, E, E], [X, E, E, E], [E, E, E, E]],
                (0, 0),
                [[O, E, E, E], [X, E, E, E], [X, E, E, E], [E, E, E, E]],
            ),
            (
                [[X, E, E, E], [E, E, E, E], [X, E, E, E], [E, E, E, E]],
                (1, 0),
                [[X, E, E, E], [O, E, E, E], [X, E, E, E], [E, E, E, E]],
            ),
            (
                [[X, O, E, E], [E, E, E, E], [E, E, E, E], [E, E, E, E]],
                None,
                [[X, O, E, E], [E, E, E, E], [E, E, E, E], [E, E, E, E]],
            ),
        ],
    )
    def test_rule(self, grid, expected_move, expected_grid):
        agent = AlgorithmicAgent(rows=len(grid), columns=len(grid[0]))
        move = agent._rule_avoid_triples_column(grid)
        assert move == expected_move
        assert grid == expected_grid


class TestRuleBalanceRow:
    @pytest.mark.parametrize(
        "grid, expected_move, expected_grid",
        [
            (
                [[X, X, E, E], [E, E, E, E], [E, E, E, E], [E, E, E, E]],
                (0, 2),
                [[X, X, O, E], [E, E, E, E], [E, E, E, E], [E, E, E, E]],
            ),
            (
                [[O, O, E, E], [E, E, E, E], [E, E, E, E], [E, E, E, E]],
                (0, 2),
                [[O, O, X, E], [E, E, E, E], [E, E, E, E], [E, E, E, E]],
            ),
            (
                [[X, O, E, E], [E, E, E, E], [E, E, E, E], [E, E, E, E]],
                None,
                [[X, O, E, E], [E, E, E, E], [E, E, E, E], [E, E, E, E]],
            ),
        ],
    )
    def test_rule(self, grid, expected_move, expected_grid):
        agent = AlgorithmicAgent(rows=len(grid), columns=len(grid[0]))
        move = agent._rule_balance_row(grid)
        assert move == expected_move
        assert grid == expected_grid


class TestRuleBalanceColumn:
    @pytest.mark.parametrize(
        "grid, expected_move, expected_grid",
        [
            (
                [[X, E, E, E], [X, E, E, E], [E, E, E, E], [E, E, E, E]],
                (2, 0),
                [[X, E, E, E], [X, E, E, E], [O, E, E, E], [E, E, E, E]],
            ),
            (
                [[O, E, E, E], [O, E, E, E], [E, E, E, E], [E, E, E, E]],
                (2, 0),
                [[O, E, E, E], [O, E, E, E], [X, E, E, E], [E, E, E, E]],
            ),
            (
                [[X, O, E, E], [E, E, E, E], [E, E, E, E], [E, E, E, E]],
                None,
                [[X, O, E, E], [E, E, E, E], [E, E, E, E], [E, E, E, E]],
            ),
        ],
    )
    def test_rule(self, grid, expected_move, expected_grid):
        agent = AlgorithmicAgent(rows=len(grid), columns=len(grid[0]))
        move = agent._rule_balance_column(grid)
        assert move == expected_move
        assert grid == expected_grid
