from __future__ import annotations

from tic_tac_logic.agents.base_agent import Agent
from tic_tac_logic.constants import E, PLACEMENT_OPTIONS, Observation, StepResult


class AlgorithmicAgent(Agent):
    """Agent that solves the board using deterministic search."""

    def reset(self) -> None:
        """Stateless agent, nothing to reset."""
        pass

    def learn(self, step_result: StepResult) -> None:  # noqa: D401
        """Algorithmic agent does not learn."""
        pass

    # Helper methods for solving
    def _is_valid(
        self,
        grid: list[list[str]],
        row: int,
        col: int,
        symbol: str,
    ) -> bool:
        rows = len(grid)
        cols = len(grid[0])
        half_row = cols // 2
        half_col = rows // 2

        # Check row counts and triple repeats
        temp_row = grid[row].copy()
        temp_row[col] = symbol
        if temp_row.count("X") > half_row or temp_row.count("O") > half_row:
            return False
        if "XXX" in "".join(temp_row) or "OOO" in "".join(temp_row):
            return False

        # Check column counts and triple repeats
        temp_col = [grid[r][col] for r in range(rows)]
        temp_col[row] = symbol
        if temp_col.count("X") > half_col or temp_col.count("O") > half_col:
            return False
        if "XXX" in "".join(temp_col) or "OOO" in "".join(temp_col):
            return False

        # Unique row constraint when row complete
        if E not in temp_row:
            for other_row_index in range(rows):
                if other_row_index == row:
                    continue
                comparison_row = grid[other_row_index]
                if E not in comparison_row and comparison_row == temp_row:
                    return False

        # Unique column constraint when column complete
        if E not in temp_col:
            for other_col_index in range(cols):
                if other_col_index == col:
                    continue
                comparison_col = [grid[r][other_col_index] for r in range(rows)]
                if E not in comparison_col and comparison_col == temp_col:
                    return False

        return True

    def _find_empty(self, grid: list[list[str]]) -> tuple[int, int] | None:
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                if cell == E:
                    return i, j
        return None

    def _solve(self, grid: list[list[str]]) -> list[list[str]] | None:
        empty = self._find_empty(grid)
        if empty is None:
            return grid
        row, col = empty
        for symbol in PLACEMENT_OPTIONS:
            if self._is_valid(grid, row, col, symbol):
                grid[row][col] = symbol
                solved = self._solve(grid)
                if solved is not None:
                    return solved
                grid[row][col] = E
        return None

    def act(self, observation: Observation) -> tuple[tuple[int, int], str]:
        grid_copy = [row.copy() for row in observation.grid]
        solved = self._solve(grid_copy)
        if solved is None:
            raise ValueError("No valid solution found")
        for i in range(self.rows):
            for j in range(self.columns):
                if observation.grid[i][j] == E:
                    return (i, j), solved[i][j]
        raise ValueError("Grid already solved")
