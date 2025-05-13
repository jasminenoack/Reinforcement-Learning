import pytest
from queens.components.grid import Grid
import numpy as np
from queens.utils import build_board_array

correct = [
    (0, 0),
    (1, 4),
    (2, 7),
    (3, 5),
    (4, 2),
    (5, 6),
    (6, 1),
    (7, 3),
]

two_in_row = [
    (0, 0),
    (0, 4),  # same row as above
    (2, 7),
    (3, 5),
    (4, 2),
    (5, 6),
    (6, 1),
    (7, 3),
]
two_in_column = [
    (0, 0),
    (1, 0),  # same column as above
    (2, 7),
    (3, 5),
    (4, 2),
    (5, 6),
    (6, 1),
    (7, 3),
]
two_on_diagonal = [
    (0, 0),
    (1, 4),
    (2, 7),
    (3, 5),
    (4, 2),  # \ diagonal clash with (5, 3)
    (5, 3),
    (6, 1),
    (7, 6),
]
two_on_reverse_diagonal = [
    (0, 7),
    (1, 6),  # same `/` diagonal as (0, 7)
    (2, 0),
    (3, 5),
    (4, 2),
    (5, 4),
    (6, 1),
    (7, 3),
]


class TestSolved:
    @pytest.mark.parametrize(
        "queens, expected",
        [
            [two_in_row, False],
            [two_in_column, False],
            [two_on_diagonal, False],
            [two_on_reverse_diagonal, False],
        ],
    )
    def test_fails_for_bad_options(
        self,
        queens: list[tuple[int, int]],
        expected: bool,
    ):
        board = build_board_array(queens)
        full_board = np.array(board)
        grid = Grid(full_board)
        assert grid.solved == expected

    def test_fails_for_more_than_8_queens(self):
        queens = [
            (0, 0),
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 4),
            (5, 5),
            (6, 6),
            (7, 7),
            (2, 3),
        ]
        board = build_board_array(queens)
        full_board = np.array(board)
        grid = Grid(full_board)
        assert grid.solved is False

    def test_fails_for_less_than_8_queens(self):
        queens = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)]
        board = build_board_array(queens)
        full_board = np.array(board)
        grid = Grid(full_board)
        assert grid.solved is False

    def test_succeeds_for_8_queens(self):
        board = build_board_array(
            [(0, 0), (1, 4), (2, 7), (3, 5), (4, 2), (5, 6), (6, 1), (7, 3)]
        )
        full_board = np.array(board)
        grid = Grid(full_board)
        assert grid.solved is True
