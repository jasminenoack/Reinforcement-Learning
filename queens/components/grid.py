import numpy as np
from numpy.typing import NDArray


class Grid:
    def __init__(self, board: NDArray[np.int_]):
        self.board = board

    @property
    def _two_in_row(self) -> bool:
        return bool(np.any(np.sum(self.board, axis=1) > 1))

    @property
    def _two_in_column(self) -> bool:
        return bool(np.any(np.sum(self.board, axis=0) > 1))

    @property
    def _two_on_diagonal(self) -> bool:
        for k in range(-7, 8):
            diag = np.diag(self.board, k)
            print(diag)
            if np.sum(diag) > 1:
                return True
        return False

    @property
    def _two_on_reverse_diagonal(self) -> bool:
        for k in range(-7, 8):
            diag = np.diag(np.fliplr(self.board), k)
            print(diag)
            if np.sum(diag) > 1:
                return True
        return False

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
