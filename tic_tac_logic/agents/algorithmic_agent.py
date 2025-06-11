from __future__ import annotations

from tic_tac_logic.agents.base_agent import Agent as BaseAgent
from tic_tac_logic.constants import X, O, E, Observation, StepResult


class AlgorithmicAgent(BaseAgent):
    """
    Deterministic agent for solving Tic-Tac-Logic puzzles.
    
    This is an agent being used as a testing ground for the codex genai coding agent. 
    
    I am aware it does not right now include all the rules, we are working on getting it to a good place 
    """

    def act(self, observation: Observation) -> tuple[tuple[int, int], str]:
        board = [row.copy() for row in observation.grid]
        move = self._deduce_one_move(board)
        if move is None:
            raise ValueError("Unable to deterministically find next move")
        row_i, col_i = move
        if observation.grid[row_i][col_i] != E:
            raise ValueError("Attempting to change a fixed value")
        return move, board[row_i][col_i]

    def reset(self) -> None:  # pragma: no cover - agent has no state
        pass

    def learn(self, step_result: StepResult) -> None:  # pragma: no cover
        pass

    def _opposite(self, symbol: str) -> str:
        if symbol not in (X, O):
            raise ValueError("_opposite called with invalid symbol")
        return O if symbol == X else X

    def _deduce_one_move(self, grid: list[list[str]]) -> tuple[int, int] | None:
        """Apply rules until exactly one cell changes or no progress is possible."""
        for rule in (
            self._rule_avoid_triples_row,
            self._rule_avoid_triples_column,
            self._rule_balance_row,
            self._rule_balance_column,
        ):
            move = rule(grid)
            if move:
                return move
        return None

    def _rule_avoid_triples_row(self, grid: list[list[str]]) -> tuple[int, int] | None:
        size = len(grid)
        for r, row in enumerate(grid):
            for c in range(size):
                if row[c] != E:
                    continue
                if c >= 2 and row[c - 1] == row[c - 2] != E:
                    row[c] = self._opposite(row[c - 1])
                    return r, c
                if c <= size - 3 and row[c + 1] == row[c + 2] != E:
                    row[c] = self._opposite(row[c + 1])
                    return r, c
                if 0 < c < size - 1 and row[c - 1] == row[c + 1] != E:
                    row[c] = self._opposite(row[c - 1])
                    return r, c
        return None

    def _rule_avoid_triples_column(
        self, grid: list[list[str]]
    ) -> tuple[int, int] | None:
        size = len(grid)
        for c in range(size):
            column = [grid[r][c] for r in range(size)]
            for r in range(size):
                if column[r] != E:
                    continue
                if r >= 2 and column[r - 1] == column[r - 2] != E:
                    grid[r][c] = self._opposite(column[r - 1])
                    return r, c
                if r <= size - 3 and column[r + 1] == column[r + 2] != E:
                    grid[r][c] = self._opposite(column[r + 1])
                    return r, c
                if 0 < r < size - 1 and column[r - 1] == column[r + 1] != E:
                    grid[r][c] = self._opposite(column[r - 1])
                    return r, c
        return None

    def _rule_balance_row(self, grid: list[list[str]]) -> tuple[int, int] | None:
        size = len(grid)
        half = size // 2
        for r, row in enumerate(grid):
            if row.count(E) == 0:
                continue
            if row.count(X) == half:
                for c in range(size):
                    if row[c] == E:
                        row[c] = O
                        return r, c
            if row.count(O) == half:
                for c in range(size):
                    if row[c] == E:
                        row[c] = X
                        return r, c
        return None

    def _rule_balance_column(self, grid: list[list[str]]) -> tuple[int, int] | None:
        size = len(grid)
        half = size // 2
        for c in range(size):
            column = [grid[r][c] for r in range(size)]
            if column.count(E) == 0:
                continue
            if column.count(X) == half:
                for r in range(size):
                    if grid[r][c] == E:
                        grid[r][c] = O
                        return r, c
            if column.count(O) == half:
                for r in range(size):
                    if grid[r][c] == E:
                        grid[r][c] = X
                        return r, c
        return None
