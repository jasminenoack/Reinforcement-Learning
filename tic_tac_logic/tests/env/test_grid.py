from tic_tac_logic.env.grid import Grid
from tic_tac_logic.sample_grids import get_one_off_grid
from tic_tac_logic.constants import X, O, E, StepResult
import pytest


class TestInit:
    def test_sets_grid(self):
        grid = Grid(get_one_off_grid())
        assert grid.grid == get_one_off_grid()

    def test_fails_if_odd_number_of_rows(self):
        with pytest.raises(ValueError, match="Grid must have an even number of rows."):
            Grid([[X, O], [O, X], [X, O]])

    def test_fails_if_rows_are_not_same_length(self):
        with pytest.raises(
            ValueError, match="All rows in the grid must have the same length."
        ):
            Grid(
                [
                    [X, O, X, O],
                    [O, X],
                    [O, X, O, X],
                    [X, O, O, X],
                ]
            )

    def test_fails_if_odd_number_of_columns(self):
        with pytest.raises(
            ValueError, match="Grid must have an even number of columns."
        ):
            Grid([[X, O, X], [O, X, O]])


class TestReset:
    def test_resets_grid(self):
        grid = Grid(get_one_off_grid())
        grid.grid[0][0] = X
        grid.reset()
        assert grid.grid == get_one_off_grid()

    def test_resets_score(self):
        grid = Grid(get_one_off_grid())
        grid.score = 100
        grid.reset()
        assert grid.score == 0

    def test_resets_actions(self):
        grid = Grid(get_one_off_grid())
        grid.actions = 10
        grid.reset()
        assert grid.actions == 0


class TestLost:
    @pytest.mark.parametrize(
        "grid, expected",
        [
            (get_one_off_grid(), False),
            # if too many X in a row
            (
                [
                    [X, X, E, X, X, E],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                ],
                True,
            ),
            # if too many O in a row
            (
                [
                    [O, O, E, E, O, O],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                ],
                True,
            ),
            # if too many X in a column
            (
                [
                    [X, E, E, E, E, E],
                    [X, E, E, E, E, E],
                    [E, E, E, E, E, E],
                    [X, E, E, E, E, E],
                    [X, E, E, E, E, E],
                    [E, E, E, E, E, E],
                ],
                True,
            ),
            # if too many O in a column
            (
                [
                    [O, E, E, E, E, E],
                    [O, E, E, E, E, E],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                    [O, E, E, E, E, E],
                    [O, E, E, E, E, E],
                ],
                True,
            ),
            # more than 2 x next to each other in a row
            (
                [
                    [X, X, X, E, E, E],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                ],
                True,
            ),
            # more than 2 o next to each other in a row
            (
                [
                    [O, O, O, E, E, E],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                ],
                True,
            ),
            # more than 2 x next to each other in a column
            (
                [
                    [X, E, E, E, E, E],
                    [X, E, E, E, E, E],
                    [X, E, E, E, E, E],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                ],
                True,
            ),
            # more than 2 o next to each other in a column
            (
                [
                    [O, E, E, E, E, E],
                    [O, E, E, E, E, E],
                    [O, E, E, E, E, E],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                ],
                True,
            ),
            # 2 rows with complete X with X identical
            (
                [
                    [X, X, E, X, E, E],
                    [X, X, E, X, E, E],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                ],
                True,
            ),
            # 2 rows with complete O with O identical
            (
                [
                    [O, O, E, O, E, E],
                    [O, O, E, O, E, E],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                ],
                True,
            ),
            # 2 columns with complete X with X identical
            (
                [
                    [X, E, X, E, E, E],
                    [X, E, X, E, E, E],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                    [X, E, X, E, E, E],
                ],
                True,
            ),
            # 2 columns with complete O with O identical
            (
                [
                    [O, E, O, E, E, E],
                    [O, E, O, E, E, E],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                    [O, E, O, E, E, E],
                ],
                True,
            ),
            # handles asymmetric grid wide
            (
                [
                    [X, E, X, E, E, E],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                ],
                False,
            ),
            # handles asymmetric grid tall
            (
                [
                    [X, E],
                    [E, E],
                    [E, E],
                    [E, E],
                ],
                False,
            ),
        ],
    )
    def test_lost(self, grid: list[list[str]], expected: bool):
        grid_instance = Grid(grid)
        assert grid_instance.lost()[0] == expected

    def test_lost_if_max_steps_reached(self):
        grid = Grid(get_one_off_grid())
        grid.actions = 1000
        assert grid.lost() == (True, "Maximum number of steps reached")


class TestWon:
    def test_did_not_win_if_not_complete(self):
        grid = Grid(get_one_off_grid())
        assert grid.won() == (False, "There are empty squares")

    def test_won_if_complete(self):
        grid = Grid(
            [
                [X, X, O, O],
                [O, O, X, X],
                [O, X, X, O],
                [X, O, O, X],
            ]
        )
        assert grid.won() == (True, "")

    def test_not_won_if_complete_but_lost(self):
        grid = Grid(
            [
                [X, X, O, O],
                [O, O, X, X],
                [O, X, X, O],
                [X, O, O, O],
            ]
        )
        assert grid.won() == (False, "TOO_MANY_O_IN_ROW")


class TestAct:
    def test_sad_if_full(self):
        grid = Grid(get_one_off_grid())
        assert grid.act((0, 0), X) == StepResult(
            coordinate=(0, 0),
            score=-10,
            symbol=X,
            loss_reason="",
            pre_step_grid=[
                ["X", "O", "X", "O"],
                ["O", " ", "O", "X"],
                ["O", "X", "X", "O"],
                ["X", "O", "O", "X"],
            ],
            grid=[
                ["X", "O", "X", "O"],
                ["O", " ", "O", "X"],
                ["O", "X", "X", "O"],
                ["X", "O", "O", "X"],
            ],
        )

    def test_updates_grid(self):
        grid = Grid(get_one_off_grid())
        result = grid.act((1, 1), X)
        assert grid.grid[1][1] == X
        assert grid.won()[0]
        assert result == StepResult(
            coordinate=(1, 1),
            score=100,
            symbol=X,
            loss_reason="",
            pre_step_grid=[
                ["X", "O", "X", "O"],
                ["O", " ", "O", "X"],
                ["O", "X", "X", "O"],
                ["X", "O", "O", "X"],
            ],
            grid=[
                ["X", "O", "X", "O"],
                ["O", "X", "O", "X"],
                ["O", "X", "X", "O"],
                ["X", "O", "O", "X"],
            ],
        )

    def test_handles_confidnet_move(self):
        grid = Grid(
            [
                [X, E, X, E, E, E],
                [E, E, E, E, E, E],
                [E, E, E, E, E, E],
                [E, E, E, E, E, E],
                [E, E, E, E, E, E],
                [E, E, E, E, E, E],
            ]
        )
        result = grid.act((0, 1), O)
        assert grid.grid[0][1] == O
        assert not grid.won()[0]
        assert result == StepResult(
            coordinate=(0, 1),
            score=10,
            symbol=O,
            loss_reason="",
            pre_step_grid=[
                ["X", " ", "X", " ", " ", " "],
                [" ", " ", " ", " ", " ", " "],
                [" ", " ", " ", " ", " ", " "],
                [" ", " ", " ", " ", " ", " "],
                [" ", " ", " ", " ", " ", " "],
                [" ", " ", " ", " ", " ", " "],
            ],
            grid=[
                ["X", "O", "X", " ", " ", " "],
                [" ", " ", " ", " ", " ", " "],
                [" ", " ", " ", " ", " ", " "],
                [" ", " ", " ", " ", " ", " "],
                [" ", " ", " ", " ", " ", " "],
                [" ", " ", " ", " ", " ", " "],
            ],
        )
        assert grid.actions == 1
        assert grid.score == 10

    def test_handles_non_confidnet_move(self):
        grid = Grid(
            [
                [X, E, X, E, E, E],
                [E, E, E, E, E, E],
                [E, E, E, E, E, E],
                [E, E, E, E, E, E],
                [E, E, E, E, E, E],
                [E, E, E, E, E, E],
            ]
        )
        result = grid.act((1, 1), O)
        assert grid.grid[1][1] == O
        assert not grid.won()[0]
        assert result == StepResult(
            coordinate=(1, 1),
            score=0,
            symbol=O,
            loss_reason="",
            pre_step_grid=[
                ["X", " ", "X", " ", " ", " "],
                [" ", " ", " ", " ", " ", " "],
                [" ", " ", " ", " ", " ", " "],
                [" ", " ", " ", " ", " ", " "],
                [" ", " ", " ", " ", " ", " "],
                [" ", " ", " ", " ", " ", " "],
            ],
            grid=[
                ["X", " ", "X", " ", " ", " "],
                [" ", "O", " ", " ", " ", " "],
                [" ", " ", " ", " ", " ", " "],
                [" ", " ", " ", " ", " ", " "],
                [" ", " ", " ", " ", " ", " "],
                [" ", " ", " ", " ", " ", " "],
            ],
        )


class TestPlacementConfidence:
    @pytest.mark.parametrize(
        "start_grid, coordinate, confidence",
        # between two of same symbol in row
        [
            (
                [
                    [X, O, X, E, E, E],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                ],
                (0, 1),
                10,
            ),
            # not if one of other symbol
            (
                [
                    [E, O, X, E, E, E],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                ],
                (0, 1),
                1,
            ),
            # not if second of same symbol
            (
                [
                    [O, O, X, E, E, E],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                ],
                (0, 1),
                1,
            ),
            # not on edge
            (
                [
                    [O, X, E, E, E, X],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                ],
                (0, 0),
                1,
            ),
            (
                [
                    [O, X, E, E, O, X],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                ],
                (0, 5),
                1,
            ),
            # between two of same symbol in column
            (
                [
                    [X, E, E, E, E, E],
                    [O, E, E, E, E, E],
                    [X, E, E, E, E, E],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                ],
                (1, 0),
                10,
            ),
            # not if one of other symbol
            (
                [
                    [E, E, E, E, E, E],
                    [O, X, E, E, E, E],
                    [X, E, E, E, E, E],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                ],
                (1, 0),
                1,
            ),
            # not if second of same symbol
            (
                [
                    [O, E, E, E, E, E],
                    [O, O, E, E, E, E],
                    [X, E, E, E, E, E],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                ],
                (1, 0),
                1,
            ),
            # not on edge
            (
                [
                    [O, E, E, E, E, E],
                    [X, O, E, E, E, E],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                    [X, E, E, E, E, E],
                ],
                (0, 0),
                1,
            ),
            (
                [
                    [O, E, E, E, E, E],
                    [E, O, E, E, E, X],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                    [O, E, E, E, E, E],
                    [X, E, E, E, E, O],
                ],
                (5, 0),
                1,
            ),
            # on right of 2 matching symbols in row
            (
                [
                    [X, X, O, E, E, E],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                ],
                (0, 2),
                10,
            ),
            # on left of 2 matching symbols in row
            (
                [
                    [E, E, O, X, X, E],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                ],
                (0, 2),
                10,
            ),
            # on top of 2 matching symbols in column
            (
                [
                    [E, X, E, E, E, E],
                    [E, O, E, E, E, E],
                    [E, O, E, E, E, E],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                ],
                (0, 1),
                10,
            ),
            # on bottom of 2 matching symbols in column
            (
                [
                    [E, E, E, E, E, E],
                    [E, O, E, E, E, E],
                    [E, O, E, E, E, E],
                    [E, X, E, E, E, E],
                    [E, E, E, E, E, E],
                    [E, E, E, E, E, E],
                ],
                (3, 1),
                10,
            ),
        ],
    )
    def test_returns_true_if_between_two_of_other_symbol_in_row(
        self, start_grid: list[list[str]], coordinate: tuple[int, int], confidence: int
    ):
        grid = Grid(start_grid)
        assert grid.placement_confidence(coordinate) == confidence

    def test_fails_if_cell_is_empty(self):
        grid = Grid(get_one_off_grid())
        with pytest.raises(ValueError, match="Cell is empty."):
            grid.placement_confidence((1, 1))
