import numpy as np
from numpy.typing import NDArray


class Grid:
    def __init__(self, board: NDArray[np.int_]):
        self.board = board

    @property
    def solved(self) -> bool:
        two_in_row = np.any(np.sum(self.board, axis=1) > 1)
        two_in_column = np.any(np.sum(self.board, axis=0) > 1)

        if two_in_row or two_in_column:
            return False

        for k in range(-7, 8):
            diag = np.diag(self.board, k)
            if np.sum(diag) > 1:
                return False

        for k in range(-7, 8):
            diag = np.diag(np.fliplr(self.board), k)
            if np.sum(diag) > 1:
                return False

        return True
