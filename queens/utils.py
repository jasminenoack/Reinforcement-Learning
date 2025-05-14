import numpy as np
from numpy.typing import NDArray


def build_board_array(queens: list[tuple[int, int]], size: int = 8) -> NDArray[np.int_]:
    board = np.zeros((size, size), dtype=np.int_)
    for row in queens:
        board[row[0]][row[1]] = 1
    return board
