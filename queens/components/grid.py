import numpy as np
from numpy.typing import NDArray

from queens.dtos import BoardState, StepResult, FailureType


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
        return bool(np.sum(self.board) == len(self.board))

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
    def score(self) -> int:
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

    def step(self, row: int, column: int) -> StepResult:
        if self.fully_played:
            raise ValueError("The board is already full.")
        self.board[row][column] = 1
        self.moves += 1
        return StepResult(action=(row, column))

    def is_queen(self, row: int, column: int) -> bool:
        return bool(self.board[row][column] == 1)

    def get_state(self) -> BoardState:
        return BoardState()

    def render(self) -> None:
        """Pretty-print the current board state."""
        for row in self.board.tolist():
            print(" ".join("Q" if cell else "." for cell in row))
        print(f"Moves: {self.moves}")
        print(f"Score: {self.score}")
        print(f"Failed: {self.failed}")
        print(f"Solved: {self.solved}")


class EarlyExitGrid(Grid):
    def __init__(self, board: NDArray[np.int_]):
        super().__init__(board)
        self._org_board = board
        self.reset()

    @property
    def score(self) -> int:
        score = super().score
        placed = int(np.sum(self.board))
        score += placed * 10
        return score

    @property
    def done(self) -> bool:
        return super().done or self.failed

    def step(self, row: int, column: int) -> StepResult:
        previous_score = self.score
        if self.failed:
            raise ValueError("The board is already failed.")
        result = super().step(row, column)
        final_score = self.score
        result.reward = final_score - previous_score

        failure_type: FailureType | None = None
        if self._two_in_row:
            failure_type = FailureType.ROW
        elif self._two_in_column:
            failure_type = FailureType.COLUMN
        elif self._two_on_diagonal:
            failure_type = FailureType.DIAGONAL
        elif self._two_on_reverse_diagonal:
            failure_type = FailureType.REVERSE_DIAGONAL
        failure_type = failure_type
        result.failure_type = failure_type
        result.board_state = BoardState(board=self.board)

        return result

    def get_state(self):
        state = super().get_state()
        state.board = np.copy(self.board)
        return state
