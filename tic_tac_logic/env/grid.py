from tic_tac_logic.constants import X, O, E, StepResult, Observation


class Grid:
    def __init__(self, grid: list[list[str]]) -> None:
        self.initial_grid = grid
        if len(grid) % 2 != 0:
            raise ValueError("Grid must have an even number of rows.")
        if not all(len(row) == len(grid[0]) for row in grid):
            raise ValueError("All rows in the grid must have the same length.")
        if len(grid[0]) % 2 != 0:
            raise ValueError("Grid must have an even number of columns.")
        self.reset()
        self.max_steps = 1000

    def reset(self) -> None:
        self.score = 0
        self.actions = 0
        self.grid = [row.copy() for row in self.initial_grid]
        self.uncertainty = 0

    def lost(self) -> tuple[bool, str]:
        row_width = len(self.grid[0])
        expected_row_count = row_width // 2
        complete_x_rows: list[tuple[int, ...]] = []
        complete_o_rows: list[tuple[int, ...]] = []
        for row in self.grid:
            if row.count(X) > expected_row_count:
                return True, "TOO_MANY_X_IN_ROW"
            if row.count(O) > expected_row_count:
                return True, "TOO_MANY_O_IN_ROW"
            if "".join([X, X, X]) in "".join(row):
                return True, "TOO_MANY_X_TOGETHER_IN_ROW"
            if "".join([O, O, O]) in "".join(row):
                return True, "TOO_MANY_O_TOGETHER_IN_ROW"
            if row.count(X) == expected_row_count:
                complete_x_rows.append(
                    tuple(i for i, cell in enumerate(row) if cell == X)
                )
            if row.count(O) == expected_row_count:
                complete_o_rows.append(
                    tuple(i for i, cell in enumerate(row) if cell == O)
                )
        if len(complete_x_rows) != len(set(complete_x_rows)):
            return True, "2 rows with complete X with X identical"
        if len(complete_o_rows) != len(set(complete_o_rows)):
            return True, "2 rows with complete O with O identical"

        column_width = len(self.grid)
        expected_column_count = column_width // 2
        complete_x_columns: list[tuple[int, ...]] = []
        complete_o_columns: list[tuple[int, ...]] = []
        for col_index in range(row_width):
            column = [
                self.grid[row_index][col_index] for row_index in range(len(self.grid))
            ]
            # if count X in column > column length / 2
            if column.count(X) > expected_column_count:
                return True, "TOO_MANY_X_IN_COLUMN"
            # # if count O in column > column length / 2
            if column.count(O) > expected_column_count:
                return True, "TOO_MANY_O_IN_COLUMN"
            # more than 2 X next to each other in a column
            if "".join([X, X, X]) in "".join(column):
                return True, "TOO_MANY_X_TOGETHER_IN_COLUMN"
            # more than 2 O next to each other in a column
            if "".join([O, O, O]) in "".join(column):
                return True, "TOO_MANY_O_TOGETHER_IN_COLUMN"
            if column.count(X) == expected_column_count:
                complete_x_columns.append(
                    tuple(i for i, cell in enumerate(column) if cell == X)
                )
            if column.count(O) == expected_column_count:
                complete_o_columns.append(
                    tuple(i for i, cell in enumerate(column) if cell == O)
                )
        if len(complete_x_columns) != len(set(complete_x_columns)):
            return True, "2 columns with complete X with X identical"
        if len(complete_o_columns) != len(set(complete_o_columns)):
            return True, "2 columns with complete O with O identical"
        if self.actions >= self.max_steps:
            return True, "Maximum number of steps reached"
        return False, ""

    def won(self) -> tuple[bool, str]:
        not_complete = any(row.count(E) > 0 for row in self.grid)
        if not_complete:
            return False, "There are empty squares"
        lost, reason = self.lost()
        if lost:
            return False, reason
        return True, ""

    def placement_confidence(self, coordinate: tuple[int, int]) -> int:
        row, col = coordinate
        full_row = self.grid[row]
        full_column = [self.grid[i][col] for i in range(len(self.grid))]
        elements_before_in_row = full_row[:col]
        elements_after_in_row = full_row[col + 1 :]
        elements_before_in_column = full_column[:row]
        elements_after_in_column = full_column[row + 1 :]
        current_element = self.grid[row][col]
        if current_element == E:
            raise ValueError("Cell is empty.")
        other_elementt = X if current_element == O else O

        previous_2_in_row = elements_before_in_row[-2:]
        previous_2_elements_in_column = elements_before_in_column[-2:]
        next_2_in_row = elements_after_in_row[:2]
        next_2_elements_in_column = elements_after_in_column[:2]
        # if there is a set of 2 next to it.
        if (
            previous_2_in_row.count(other_elementt) == 2
            or next_2_in_row.count(other_elementt) == 2
            or previous_2_elements_in_column.count(other_elementt) == 2
            or next_2_elements_in_column.count(other_elementt) == 2
        ):
            return 10

        either_side_in_row = elements_before_in_row[-1:] + elements_after_in_row[:1]
        either_side_in_column = (
            elements_before_in_column[-1:] + elements_after_in_column[:1]
        )
        # if there is a differing on either side
        if (
            either_side_in_row.count(other_elementt) == 2
            or either_side_in_column.count(other_elementt) == 2
        ):
            return 10

        self.uncertainty += 1
        return 1

    def act(self, coordinate: tuple[int, int], symbol: str) -> StepResult:
        pre_move_grid = [row.copy() for row in self.grid]
        row, col = coordinate
        if self.grid[row][col] != E:
            result = StepResult(
                coordinate=coordinate,
                score=-10,
                symbol=symbol,
                grid=self.grid,
                pre_step_grid=pre_move_grid,
            )
        else:
            self.grid[row][col] = symbol
            lost = self.lost()
            won = self.won()
            if lost[0]:
                result = StepResult(
                    coordinate=coordinate,
                    score=-10,
                    symbol=symbol,
                    loss_reason=lost[1],
                    grid=self.grid,
                    pre_step_grid=pre_move_grid,
                )
            elif won[0]:
                result = StepResult(
                    coordinate=coordinate,
                    score=100,
                    symbol=symbol,
                    grid=self.grid,
                    pre_step_grid=pre_move_grid,
                )
            else:
                result = StepResult(
                    coordinate=coordinate,
                    score=self.placement_confidence(coordinate),
                    symbol=symbol,
                    grid=self.grid,
                    pre_step_grid=pre_move_grid,
                )
        if result.score < 0:
            result.score = min(result.score + self.uncertainty, 0)
        else:
            result.score = max(result.score - self.uncertainty, 0)
        # if self.uncertainty > 0:
        #     result.score = 0
        self.score += result.score
        self.actions += 1
        return result

    def get_observation(self) -> Observation:
        return Observation(grid=self.grid)
