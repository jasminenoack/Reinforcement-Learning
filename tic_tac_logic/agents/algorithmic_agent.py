from typing import Optional
from tic_tac_logic.agents.base_agent import Agent
from tic_tac_logic.constants import StepResult, Observation, X, O, E


class NoValidMoveError(Exception):
    """Raised when the agent cannot find a valid move without guessing."""

    pass


class AlgorithmicAgent(Agent):
    """
    An algorithmic agent that uses logical rules to solve Binaro/tic-tac-logic puzzles.

    Rules implemented:
    1. No more than two consecutive identical symbols in any row or column
    2. Equal number of X's and O's in each completed row and column
    3. All rows must be unique, all columns must be unique

    The agent only makes moves it is certain about and raises NoValidMoveError
    if it cannot determine a move without guessing.
    """

    def __init__(self, rows: int, columns: int) -> None:
        super().__init__(rows, columns)
        # For Binaro puzzles, each row/column should have equal X's and O's
        if self.rows % 2 != 0 or self.columns % 2 != 0:
            raise ValueError("Binaro puzzles require even dimensions")
        self.target_count = (
            self.rows // 2
        )  # Target count for each symbol per row/column

    def act(self, observation: Observation) -> tuple[tuple[int, int], str]:
        """
        Find the next logical move based on Binaro rules.

        Raises NoValidMoveError if no certain move can be determined.
        """
        grid = observation.grid

        # Try each rule in order of confidence
        move = self._find_consecutive_prevention_move(grid)
        if move:
            return move

        move = self._find_completion_move(grid)
        if move:
            return move

        move = self._find_uniqueness_move(grid)
        if move:
            return move

        # If no certain move found, raise error
        raise NoValidMoveError("Cannot determine a certain move without guessing")

    def _find_consecutive_prevention_move(
        self, grid: list[list[str]]
    ) -> Optional[tuple[tuple[int, int], str]]:
        """
        Find moves that prevent three consecutive identical symbols.
        This is the highest confidence rule.
        """
        # Check rows
        for row_idx in range(self.rows):
            for col_idx in range(self.columns - 2):
                # Check for patterns like XX_ or _XX or X_X
                symbols = [grid[row_idx][col_idx + i] for i in range(3)]
                move = self._analyze_consecutive_pattern(symbols, (row_idx, col_idx))
                if move:
                    return move

        # Check columns
        for col_idx in range(self.columns):
            for row_idx in range(self.rows - 2):
                # Check for patterns like XX_ or _XX or X_X
                symbols = [grid[row_idx + i][col_idx] for i in range(3)]
                move = self._analyze_consecutive_pattern(
                    symbols, (row_idx, col_idx), is_column=True
                )
                if move:
                    return move

        return None

    def _analyze_consecutive_pattern(
        self, symbols: list[str], start_pos: tuple[int, int], is_column: bool = False
    ) -> Optional[tuple[tuple[int, int], str]]:
        """Analyze a 3-symbol pattern and return the required move if any."""
        # Pattern XX_ -> place opposite of X
        if symbols[0] == symbols[1] and symbols[0] in [X, O] and symbols[2] == E:
            opposite = O if symbols[0] == X else X
            if is_column:
                return ((start_pos[0] + 2, start_pos[1]), opposite)
            else:
                return ((start_pos[0], start_pos[1] + 2), opposite)

        # Pattern _XX -> place opposite of X
        if symbols[1] == symbols[2] and symbols[1] in [X, O] and symbols[0] == E:
            opposite = O if symbols[1] == X else X
            if is_column:
                return ((start_pos[0], start_pos[1]), opposite)
            else:
                return ((start_pos[0], start_pos[1]), opposite)

        # Pattern X_X -> place opposite of X
        if symbols[0] == symbols[2] and symbols[0] in [X, O] and symbols[1] == E:
            opposite = O if symbols[0] == X else X
            if is_column:
                return ((start_pos[0] + 1, start_pos[1]), opposite)
            else:
                return ((start_pos[0], start_pos[1] + 1), opposite)

        return None

    def _find_completion_move(
        self, grid: list[list[str]]
    ) -> Optional[tuple[tuple[int, int], str]]:
        """
        Find moves based on completing rows/columns that have reached their symbol limit.
        """
        # Check rows
        for row_idx in range(self.rows):
            row = grid[row_idx]
            x_count = row.count(X)
            o_count = row.count(O)
            empty_count = row.count(E)

            if empty_count > 0:
                if x_count == self.target_count:
                    # Fill all empty cells with O
                    for col_idx, cell in enumerate(row):
                        if cell == E:
                            return ((row_idx, col_idx), O)
                elif o_count == self.target_count:
                    # Fill all empty cells with X
                    for col_idx, cell in enumerate(row):
                        if cell == E:
                            return ((row_idx, col_idx), X)

        # Check columns
        for col_idx in range(self.columns):
            column = [grid[row_idx][col_idx] for row_idx in range(self.rows)]
            x_count = column.count(X)
            o_count = column.count(O)
            empty_count = column.count(E)

            if empty_count > 0:
                if x_count == self.target_count:
                    # Fill all empty cells with O
                    for row_idx, cell in enumerate(column):
                        if cell == E:
                            return ((row_idx, col_idx), O)
                elif o_count == self.target_count:
                    # Fill all empty cells with X
                    for row_idx, cell in enumerate(column):
                        if cell == E:
                            return ((row_idx, col_idx), X)

        return None

    def _find_uniqueness_move(
        self, grid: list[list[str]]
    ) -> Optional[tuple[tuple[int, int], str]]:
        """
        Find moves based on ensuring row/column uniqueness.
        This checks if a placement would create a duplicate row or column.
        """
        empty_cells = self._get_empty_cells(grid)

        for row, col in empty_cells:
            for symbol in [X, O]:
                # Test placement
                test_grid = [row_list[:] for row_list in grid]
                test_grid[row][col] = symbol

                # Check if this placement would create a duplicate row
                current_row = test_grid[row]
                if E not in current_row:  # Row is complete
                    # Check against all other complete rows
                    for other_row_idx, other_row in enumerate(test_grid):
                        if other_row_idx != row and E not in other_row:
                            if current_row == other_row:
                                # This placement would create duplicate rows
                                # So we must place the opposite symbol
                                opposite = O if symbol == X else X
                                if self._is_valid_placement(grid, row, col, opposite):
                                    return ((row, col), opposite)

                # Check if this placement would create a duplicate column
                current_col = [test_grid[r][col] for r in range(self.rows)]
                if E not in current_col:  # Column is complete
                    # Check against all other complete columns
                    for other_col_idx in range(self.columns):
                        if other_col_idx != col:
                            other_col = [
                                test_grid[r][other_col_idx] for r in range(self.rows)
                            ]
                            if E not in other_col and current_col == other_col:
                                # This placement would create duplicate columns
                                # So we must place the opposite symbol
                                opposite = O if symbol == X else X
                                if self._is_valid_placement(grid, row, col, opposite):
                                    return ((row, col), opposite)

        return None

    def reset(self) -> None:
        """Reset the agent's state."""
        # This agent is stateless, so no reset needed
        pass

    def learn(self, step_result: StepResult) -> None:
        """
        This agent doesn't learn from experience as it uses fixed algorithmic rules.
        """
        # Algorithmic agent doesn't learn
        pass

    def _get_empty_cells(self, grid: list[list[str]]) -> list[tuple[int, int]]:
        """Get all empty cell coordinates."""
        empty_cells = []
        for row_idx in range(self.rows):
            for col_idx in range(self.columns):
                if grid[row_idx][col_idx] == E:
                    empty_cells.append((row_idx, col_idx))
        return empty_cells

    def _is_valid_placement(
        self, grid: list[list[str]], row: int, col: int, symbol: str
    ) -> bool:
        """
        Check if placing a symbol at the given position would be valid.
        This is a helper method for more complex rule checking.
        """
        # Create a copy of the grid with the proposed placement
        test_grid = [row[:] for row in grid]
        test_grid[row][col] = symbol

        # Check if this placement violates any rules
        return self._check_no_three_consecutive(
            test_grid, row, col
        ) and self._check_count_limits(test_grid, row, col)

    def _check_no_three_consecutive(
        self, grid: list[list[str]], row: int, col: int
    ) -> bool:
        """Check if the placement creates three consecutive symbols."""
        symbol = grid[row][col]

        # Check horizontal
        for start_col in range(max(0, col - 2), min(self.columns - 2, col + 1)):
            if all(grid[row][start_col + i] == symbol for i in range(3)):
                return False

        # Check vertical
        for start_row in range(max(0, row - 2), min(self.rows - 2, row + 1)):
            if all(grid[start_row + i][col] == symbol for i in range(3)):
                return False

        return True

    def _check_count_limits(self, grid: list[list[str]], row: int, col: int) -> bool:
        """Check if the placement exceeds the symbol count limit for the row/column."""
        symbol = grid[row][col]

        # Check row count
        row_count = sum(1 for cell in grid[row] if cell == symbol)
        if row_count > self.target_count:
            return False

        # Check column count
        col_count = sum(1 for r in range(self.rows) if grid[r][col] == symbol)
        if col_count > self.target_count:
            return False

        return True

    def debug_grid_state(self, grid: list[list[str]]) -> str:
        """
        Return a string representation of the grid state for debugging.
        Shows counts and identifies potential moves.
        """
        lines = []
        lines.append("Grid Analysis:")

        # Show the grid
        for i, row in enumerate(grid):
            line = f"Row {i}: {' '.join(cell if cell != E else '_' for cell in row)}"
            x_count = row.count(X)
            o_count = row.count(O)
            e_count = row.count(E)
            line += f" (X:{x_count}, O:{o_count}, E:{e_count})"
            lines.append(line)

        lines.append("")
        lines.append("Column Analysis:")
        for j in range(self.columns):
            col = [grid[i][j] for i in range(self.rows)]
            line = f"Col {j}: {' '.join(cell if cell != E else '_' for cell in col)}"
            x_count = col.count(X)
            o_count = col.count(O)
            e_count = col.count(E)
            line += f" (X:{x_count}, O:{o_count}, E:{e_count})"
            lines.append(line)

        return "\n".join(lines)
