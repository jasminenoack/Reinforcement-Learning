import numpy as np
from numpy.typing import NDArray

from queens.dtos import BoardState


class Grid:
    def __init__(self, board: NDArray[np.int_]):
        self._org_board = board
        self.reset()

    @property
    def _values_in_rows(self) -> NDArray[np.int_]:
        return np.sum(self.board, axis=1)

    @property
    def _two_in_row(self) -> bool:
        return bool(np.any(self._values_in_rows > 1))

    @property
    def _values_in_columns(self) -> NDArray[np.int_]:
        return np.sum(self.board, axis=0)

    @property
    def _two_in_column(self) -> bool:
        return bool(np.any(self._values_in_columns > 1))

    @property
    def values_in_diagonals(self) -> NDArray[np.int_]:
        return np.array([np.sum(np.diag(self.board, k)) for k in range(-7, 8)])

    @property
    def _two_on_diagonal(self) -> bool:
        return bool(np.any(self.values_in_diagonals > 1))

    @property
    def _values_in_reverse_diagonals(self) -> NDArray[np.int_]:
        return np.array(
            [np.sum(np.diag(np.fliplr(self.board), k)) for k in range(-7, 8)]
        )

    @property
    def _two_on_reverse_diagonal(self) -> bool:
        return bool(np.any(self._values_in_reverse_diagonals > 1))

    @property
    def solved(self) -> bool:
        if (
            self._two_in_row
            or self._two_in_column
            or self._two_on_diagonal
            or self._two_on_reverse_diagonal
        ):
            return False
        return self.fully_played

    @property
    def fully_played(self) -> bool:
        return bool(np.sum(self.board) == 8)

    @property
    def done(self) -> bool:
        return self.fully_played

    @property
    def failed(self) -> bool:
        if (
            self._two_in_row
            or self._two_in_column
            or self._two_on_diagonal
            or self._two_on_reverse_diagonal
        ):
            return True
        return False

    @property
    def simple_score(self) -> int:
        """
        If you have solved + 100 points
        If you have failed - 100 points
        If you are not complete 0 points
        """
        if self.solved:
            return 100
        elif self.failed:
            return -100
        else:
            return 0

    def reset(self):
        self.moves = 0
        self.board = np.copy(self._org_board)

    def step(self, row: int, column: int):
        if self.fully_played:
            raise ValueError("The board is already full.")
        self.board[row][column] = 1
        self.moves += 1

    def is_queen(self, row: int, column: int) -> bool:
        return bool(self.board[row][column] == 1)

    def get_state(self) -> BoardState:
        return BoardState()

    def render(self):
        for row in self.board:
            print(" ".join("Q" if cell else "." for cell in row))
        print(f"Moves: {self.moves}")
        print(f"Score: {self.simple_score}")
        print(f"Failed: {self.failed}")
        print(f"Solved: {self.solved}")
