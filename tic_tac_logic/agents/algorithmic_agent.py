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
        return self._rule_avoid_triples(grid, is_row=True)

    def _rule_avoid_triples_column(self, grid: list[list[str]]) -> tuple[int, int] | None:
        return self._rule_avoid_triples(grid, is_row=False)

    def _rule_avoid_triples(self, grid: list[list[str]], is_row: bool) -> tuple[int, int] | None:
        size = len(grid)
        for i in range(size):
            line = grid[i] if is_row else [grid[j][i] for j in range(size)]
            for j in range(size):
                if line[j] != E:
                    continue
                if j >= 2 and line[j - 1] == line[j - 2] != E:
                    line[j] = self._opposite(line[j - 1])
                    return (i, j) if is_row else (j, i)
                if j <= size - 3 and line[j + 1] == line[j + 2] != E:
                    line[j] = self._opposite(line[j + 1])
                    return (i, j) if is_row else (j, i)
                if 0 < j < size - 1 and line[j - 1] == line[j + 1] != E:
                    line[j] = self._opposite(line[j - 1])
                    return (i, j) if is_row else (j, i)
        return None

    def _rule_balance_row(self, grid: list[list[str]]) -> tuple[int, int] | None:
        return self._rule_balance(grid, is_row=True)

    def _rule_balance_column(self, grid: list[list[str]]) -> tuple[int, int] | None:
        return self._rule_balance(grid, is_row=False)

    def _rule_balance(self, grid: list[list[str]], is_row: bool) -> tuple[int, int] | None:
        size = len(grid)
        half = size // 2
        for i in range(size):
            line = grid[i] if is_row else [grid[j][i] for j in range(size)]
            if line.count(E) == 0:
                continue
            if line.count(X) == half:
                for j in range(size):
                    if line[j] == E:
                        line[j] = O
                        return (i, j) if is_row else (j, i)
            if line.count(O) == half:
                for j in range(size):
                    if line[j] == E:
                        line[j] = X
                        return (i, j) if is_row else (j, i)
        return None
