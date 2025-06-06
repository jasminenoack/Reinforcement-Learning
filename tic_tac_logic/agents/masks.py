from dataclasses import dataclass
from tic_tac_logic.constants import E


@dataclass(frozen=True)
class MaskKey:
    mask_type: "AbstractMask"
    pattern: str
    symbol: str


@dataclass(frozen=True)
class MaskRules:
    rows_above: int
    rows_below: int
    columns_left: int
    columns_right: int

    def get_pattern(
        self, coord: tuple[int, int], grid: list[list[str]]
    ) -> list[list[str]] | None:
        row_i, col_i = coord

        first_row = row_i - self.rows_above
        if first_row < 0:
            return None
        last_row = row_i + self.rows_below
        if last_row >= len(grid):
            return None
        first_col = col_i - self.columns_left
        if first_col < 0:
            return None
        last_col = col_i + self.columns_right
        if last_col >= len(grid[0]):
            return None

        rows = grid[first_row : last_row + 1]
        mini_grid = [row[first_col : last_col + 1] for row in rows]
        return mini_grid


class AbstractMask:
    def __init__(self, match_symbol: str | None, rule: MaskRules) -> None:
        self.match_symbol = match_symbol
        self.rule = rule
        row_len = rule.rows_above + rule.rows_below + 1
        column_len = rule.columns_left + rule.columns_right + 1
        row_position = rule.rows_above
        column_position = rule.columns_left
        position = (row_position, column_position)
        self.name = f"Mask|{row_len}x{column_len}|{position}"
        if match_symbol:
            self.name += f"|{match_symbol}"

    def create_mask_key(self, value: list[list[str]], current: str) -> MaskKey:
        rows = ["".join(row).replace(" ", "_") for row in value]
        return MaskKey(
            mask_type=self,
            pattern=("\n").join(rows),
            symbol=current,
        )

    def remove_non_matching(self, grid: list[list[str]]) -> list[list[str]]:
        return [
            [cell if cell == self.match_symbol else E for cell in row] for row in grid
        ]

    def get_mask(
        self, coord: tuple[int, int], grid: list[list[str]], current: str
    ) -> MaskKey | None:
        if self.match_symbol:
            grid = self.remove_non_matching(grid)
        section = self.rule.get_pattern(coord, grid)
        if section is None:
            return None
        return self.create_mask_key(section, current=current)

    def __eq__(self, value: "AbstractMask") -> bool:
        return self.match_symbol == value.match_symbol and self.rule == value.rule

    def __hash__(self) -> int:
        return (self.match_symbol, self.rule).__hash__()

    def __str__(self) -> str:
        return f"{self.name}"

    def __repr__(self):
        return str(self)


def print_mask(mask: AbstractMask):
    rows_above = mask.rule.rows_above
    rows_below = mask.rule.rows_below
    columns_left = mask.rule.columns_left
    columns_right = mask.rule.columns_right
    total_columns = columns_left + columns_right + 1
    total_rows = rows_above + rows_below + 1
    for row in range(total_rows):
        for column in range(total_columns):
            if row == rows_above and column == columns_left:
                print("+", end="")
            else:
                print("_", end="")
        print()
    print()


def generate_pool_masks(
    rows: int,
    columns: int,
    debug: bool = False,
    skip_rows_under: int = 0,
    skip_columns_under: int = 0,
) -> list[AbstractMask]:
    if debug:
        print(rows, columns)
    masks: list[AbstractMask] = []
    for rows in range(skip_rows_under, rows + 1):
        if debug:
            print("   ", rows)
        for columns in range(skip_columns_under, columns + 1):
            if debug:
                print("      ", columns)
            for current_row in range(rows):
                for current_column in range(columns):
                    for symbol in ["X", "O", None]:
                        if debug:
                            print(
                                f"{rows}X{columns} ({current_row},{current_column}) {symbol}"
                            )
                        rows_above = current_row
                        rows_below = rows - current_row - 1
                        columns_left = current_column
                        columns_right = columns - current_column - 1
                        masks.append(
                            AbstractMask(
                                match_symbol=symbol,
                                rule=MaskRules(
                                    rows_above=rows_above,
                                    rows_below=rows_below,
                                    columns_left=columns_left,
                                    columns_right=columns_right,
                                ),
                            )
                        )
                    if debug:
                        print_mask(masks[-1])
    return masks


if __name__ == "__main__":
    masks = generate_pool_masks(3, 3, debug=True)
