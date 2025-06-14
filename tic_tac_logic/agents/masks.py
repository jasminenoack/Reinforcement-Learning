from dataclasses import dataclass
from typing import Generator
from tic_tac_logic.constants import E, X, O, PLACEMENT_OPTIONS


@dataclass(frozen=True)
class MaskRules:
    rows_above: int
    rows_below: int
    columns_left: int
    columns_right: int

    def rows(self) -> int:
        return self.rows_above + self.rows_below + 1

    def columns(self) -> int:
        return self.columns_left + self.columns_right + 1

    def total_cells(self) -> int:
        return self.rows() * self.columns()

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


class AbstractMaskFactory:
    def __init__(
        self, match_symbol: str | None, rule: MaskRules, debug: bool = False
    ) -> None:
        self.match_symbol = match_symbol
        self.rule = rule
        row_len = rule.rows_above + rule.rows_below + 1
        column_len = rule.columns_left + rule.columns_right + 1
        row_position = rule.rows_above
        column_position = rule.columns_left
        position = (row_position, column_position)
        self.row_position = row_position
        self.column_position = column_position
        self.name = f"Mask|{row_len}x{column_len}|{position}"
        if match_symbol:
            self.name += f"|{match_symbol}"
        self.debug = debug

    def remove_non_matching(self, grid: list[list[str]]) -> list[list[str]]:
        return [
            [cell if cell == self.match_symbol else E for cell in row] for row in grid
        ]

    def __eq__(self, value: "AbstractMaskFactory") -> bool:
        return self.match_symbol == value.match_symbol and self.rule == value.rule

    def __hash__(self) -> int:
        return (self.match_symbol, self.rule).__hash__()

    def __str__(self) -> str:
        return f"{self.name}"

    def __repr__(self):
        return str(self)

    def rows(self) -> int:
        return self.rule.rows()

    def columns(self) -> int:
        return self.rule.columns()

    def total_cells(self) -> int:
        return self.rule.total_cells()

    def generate_masks(self) -> list["CompleteMask"]:
        masks: list[CompleteMask] = []
        symbols = [E]
        if self.match_symbol:
            symbols.append(self.match_symbol)
        else:
            symbols.append(X)
            symbols.append(O)

        patterns: list[list[str]] = [[]]
        total_cells = self.total_cells()
        current_cell = self.row_position * self.columns() + self.column_position

        for i in range(total_cells):
            if i == current_cell:
                patterns = [pattern + [E] for pattern in patterns]
            else:
                patterns = [
                    pattern + [symbol] for pattern in patterns for symbol in symbols
                ]

        assert all(
            len(pattern) == total_cells for pattern in patterns
        ), "All patterns must have the same length as total cells"
        assert all(pattern[current_cell] == E for pattern in patterns)

        for symbol in PLACEMENT_OPTIONS:
            masks += [
                CompleteMask(
                    rules=self.rule,
                    pattern="".join(pattern).replace(" ", "_"),
                    symbol_to_place=symbol,
                    match_symbol=self.match_symbol,
                )
                for pattern in patterns
            ]

        return masks


def print_mask(mask: "CompleteMask"):
    rows_above = mask.rules.rows_above
    rows_below = mask.rules.rows_below
    columns_left = mask.rules.columns_left
    columns_right = mask.rules.columns_right
    total_columns = columns_left + columns_right + 1
    total_rows = rows_above + rows_below + 1
    for row in range(total_rows):
        for column in range(total_columns):
            if row == rows_above and column == columns_left:
                print("+", end="")
            else:
                current_index = row * total_columns + column
                print(mask.pattern[current_index].replace(" ", "_"), end="")
        print()
    print()


@dataclass(frozen=True)
class CompleteMask:
    rules: MaskRules
    pattern: str
    symbol_to_place: str
    match_symbol: str | None

    def __repr__(self):
        return f"CompleteMask(({self.rules.rows()},{self.rules.columns()}) - {self.pattern} <{self.match_symbol}> <- {self.symbol_to_place})"

    @property
    def name(self) -> str:
        row_len = self.rules.rows_above + self.rules.rows_below + 1
        column_len = self.rules.columns_left + self.rules.columns_right + 1
        position = (self.rules.rows_above, self.rules.columns_left)
        name = f"Mask|{row_len}x{column_len}|{position}"
        if self.match_symbol:
            name += f"|<{self.match_symbol}>"
        name += f"|{self.symbol_to_place}"
        return name

    def _mask_matches_pattern(self, pattern: list[list[str]]) -> bool:
        grid_pattern = "".join([item for row in pattern for item in row]).replace(
            " ", "_"
        )
        mask_pattern = self.pattern.replace(" ", "_")
        return grid_pattern == mask_pattern

    def _mask_matches_symbol(self, current: str) -> bool:
        return self.symbol_to_place == current

    def remove_non_matching(self, grid: list[list[str]]) -> list[list[str]]:
        return [
            [cell if cell == self.match_symbol else E for cell in row] for row in grid
        ]

    def mask_applies(
        self, coord: tuple[int, int], grid: list[list[str]], current: str
    ) -> bool:
        if not self._mask_matches_symbol(current):
            return False

        if self.match_symbol:
            grid = self.remove_non_matching(grid)
        section = self.rules.get_pattern(coord, grid)
        return bool(section) and self._mask_matches_pattern(section)

    def total_cells(self) -> int:
        return self.rules.total_cells()


@dataclass(frozen=True)
class _MaskIdentifier:
    rules: MaskRules
    pattern: str
    current_symbol: str


def generate_pool_masks(
    rows: int,
    columns: int,
    debug: bool = False,
    skip_rows_under: int = 0,
    skip_columns_under: int = 0,
) -> Generator[CompleteMask, None, None]:
    seen: set[_MaskIdentifier] = set()
    if debug:
        print(rows, columns)

    sizes: list[tuple[int, int]] = []
    for rows in range(skip_rows_under, rows + 1):
        for columns in range(skip_columns_under, columns + 1):
            sizes = sizes + [(rows, columns)]
    sizes = sorted(sizes, key=lambda x: sum(x))
    for rows, columns in sizes:
        entry_location_sets: list[tuple[int, int]] = []
        for entry_row in range(rows):
            for entry_column in range(columns):
                entry_location_sets.append((entry_row, entry_column))
        entry_location_sets = sorted(entry_location_sets, key=lambda x: sum(x))
        for entry_row, entry_column in entry_location_sets:
            for symbol in ["X", "O", None]:
                if debug:
                    print(f"{rows}X{columns} ({entry_row},{entry_column}) {symbol}")
                rows_above = entry_row
                rows_below = rows - entry_row - 1
                columns_left = entry_column
                columns_right = columns - entry_column - 1
                next_masks = AbstractMaskFactory(
                    match_symbol=symbol,
                    rule=MaskRules(
                        rows_above=rows_above,
                        rows_below=rows_below,
                        columns_left=columns_left,
                        columns_right=columns_right,
                    ),
                ).generate_masks()
                for mask in next_masks:
                    unique_key = _MaskIdentifier(
                        rules=mask.rules,
                        pattern=mask.pattern,
                        current_symbol=mask.symbol_to_place,
                    )
                    if unique_key in seen:
                        continue
                    seen.add(unique_key)
                    yield mask
                if debug:
                    print(len(next_masks), "masks generated in iteration")
                    print_mask(next_masks[-1])


@dataclass(frozen=True)
class _MemoizedMaskKey:
    coordinate: tuple[int, int]
    grid: tuple[tuple[str, ...], ...]


_memoized_patterns: dict[_MemoizedMaskKey, set[str]] = {}


def generate_all_patterns(
    coordinate: tuple[int, int],
    grid: list[list[str]],
):
    memoized_key = _MemoizedMaskKey(
        coordinate=coordinate,
        grid=tuple(tuple(row) for row in grid),
    )
    if memoized_key in _memoized_patterns:
        return _memoized_patterns[memoized_key]

    patterns: set[str] = set()
    for rows_included in range(1, len(grid) + 1):
        for columns_included in range(1, len(grid[0]) + 1):
            for current_row in range(rows_included):
                for current_column in range(columns_included):
                    rows_above = current_row
                    rows_below = rows_included - current_row - 1
                    columns_left = current_column
                    columns_right = columns_included - current_column - 1
                    rules = MaskRules(
                        rows_above=rows_above,
                        rows_below=rows_below,
                        columns_left=columns_left,
                        columns_right=columns_right,
                    )
                    pattern = rules.get_pattern(coordinate, grid)
                    if pattern:
                        patterns.add(
                            "".join(item for row in pattern for item in row).replace(
                                " ", "_"
                            )
                        )
                        patterns.add(
                            "".join(item for row in pattern for item in row)
                            .replace(" ", "_")
                            .replace("X", "_")
                        )
                        patterns.add(
                            "".join(item for row in pattern for item in row)
                            .replace(" ", "_")
                            .replace("O", "_")
                        )
    _memoized_patterns[memoized_key] = patterns
    return patterns


if __name__ == "__main__":
    masks = generate_pool_masks(2, 1, debug=True)
