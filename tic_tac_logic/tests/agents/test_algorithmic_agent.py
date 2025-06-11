import pytest

from tic_tac_logic.agents.algorithmic_agent import AlgorithmicAgent
from tic_tac_logic.constants import X, O, E
from tic_tac_logic.env.grid import Grid
from tic_tac_logic.sample_grids import get_testing_sample_grids


class TestOpposite:
    @pytest.mark.parametrize("symbol,expected", [(X, O), (O, X)])
    def test_returns_opposite_symbol(self, symbol: str, expected: str) -> None:
        agent = AlgorithmicAgent(rows=2, columns=2)
        assert agent._opposite(symbol) == expected

    def test_raises_error_on_invalid_symbol(self) -> None:
        agent = AlgorithmicAgent(rows=2, columns=2)
        with pytest.raises(ValueError):
            agent._opposite(E)


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
    def test_rule(
        self,
        grid: list[list[str]],
        expected_move: tuple[int, int] | None,
        expected_grid: list[list[str]],
    ) -> None:
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
    def test_rule(
        self,
        grid: list[list[str]],
        expected_move: tuple[int, int] | None,
        expected_grid: list[list[str]],
    ) -> None:
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
    def test_rule(
        self,
        grid: list[list[str]],
        expected_move: tuple[int, int] | None,
        expected_grid: list[list[str]],
    ) -> None:
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
    def test_rule(
        self,
        grid: list[list[str]],
        expected_move: tuple[int, int] | None,
        expected_grid: list[list[str]],
    ) -> None:
        agent = AlgorithmicAgent(rows=len(grid), columns=len(grid[0]))
        move = agent._rule_balance_column(grid)
        assert move == expected_move
        assert grid == expected_grid


class TestAlgorithmicAgent:
    def test_solves_first_sample_grid(self) -> None:
        puzzle = get_testing_sample_grids()[0]
        grid = Grid(puzzle.grid)
        agent = AlgorithmicAgent(rows=len(puzzle.grid), columns=len(puzzle.grid[0]))

        while not grid.won()[0]:
            coord, symbol = agent.act(grid.get_observation())
            result = grid.act(coord, symbol)
            agent.learn(result)
            assert not result.loss_reason
        assert grid.grid == puzzle.expected

    def test_solves_all_sample_grids(self) -> None:
        for puzzle in get_testing_sample_grids():
            grid = Grid(puzzle.grid)
            agent = AlgorithmicAgent(rows=len(puzzle.grid), columns=len(puzzle.grid[0]))
            while not grid.won()[0]:
                coord, symbol = agent.act(grid.get_observation())
                result = grid.act(coord, symbol)
                agent.learn(result)
                assert not result.loss_reason
            assert grid.won()[0]
