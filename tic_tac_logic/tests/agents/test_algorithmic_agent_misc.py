from tic_tac_logic.agents.algorithmic_agent import AlgorithmicAgent
from tic_tac_logic.constants import X, O, E


class TestOpposite:
    def test_returns_opposite_symbol(self) -> None:
        agent = AlgorithmicAgent(rows=2, columns=2)
        assert agent._opposite(X) == O
        assert agent._opposite(O) == X
        assert agent._opposite(E) == X


class TestDeduceOneMove:
    def test_prefers_row_rule_over_column_rule(self) -> None:
        grid = [
            [X, X, E, E],
            [X, E, E, E],
            [E, E, E, E],
            [E, E, E, E],
        ]
        expected = [
            [X, X, O, E],
            [X, E, E, E],
            [E, E, E, E],
            [E, E, E, E],
        ]
        agent = AlgorithmicAgent(rows=4, columns=4)
        move = agent._deduce_one_move(grid)
        assert move == (0, 2)
        assert grid == expected

    def test_uses_column_rule_when_row_rule_not_applicable(self) -> None:
        grid = [
            [X, E, E, E],
            [X, E, E, E],
            [E, E, E, E],
            [E, E, E, E],
        ]
        expected = [
            [X, E, E, E],
            [X, E, E, E],
            [O, E, E, E],
            [E, E, E, E],
        ]
        agent = AlgorithmicAgent(rows=4, columns=4)
        move = agent._deduce_one_move(grid)
        assert move == (2, 0)
        assert grid == expected

    def test_uses_row_balance_when_other_rules_fail(self) -> None:
        grid = [
            [X, O, X, E],
            [O, X, O, E],
            [E, E, E, E],
            [E, E, E, E],
        ]
        expected = [
            [X, O, X, O],
            [O, X, O, E],
            [E, E, E, E],
            [E, E, E, E],
        ]
        agent = AlgorithmicAgent(rows=4, columns=4)
        move = agent._deduce_one_move(grid)
        assert move == (0, 3)
        assert grid == expected

    def test_uses_column_balance_as_last_resort(self) -> None:
        grid = [
            [X, O, E, E],
            [X, E, E, E],
            [O, E, E, E],
            [E, E, E, E],
        ]
        expected = [
            [X, O, E, E],
            [X, E, E, E],
            [O, E, E, E],
            [O, E, E, E],
        ]
        agent = AlgorithmicAgent(rows=4, columns=4)
        move = agent._deduce_one_move(grid)
        assert move == (3, 0)
        assert grid == expected

    def test_returns_none_when_no_rule_applies(self) -> None:
        grid = [
            [X, O, E, E],
            [O, X, E, E],
            [E, E, X, O],
            [E, E, O, X],
        ]
        expected = [row.copy() for row in grid]
        agent = AlgorithmicAgent(rows=4, columns=4)
        move = agent._deduce_one_move(grid)
        assert move is None
        assert grid == expected
