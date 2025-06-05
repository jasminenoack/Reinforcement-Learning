from dataclasses import dataclass
from tic_tac_logic.constants import E


@dataclass(frozen=True)
class MaskKey:
    mask_type: type["AbstractMask"]
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


@dataclass
class AbstractMask:
    match_symbol: str | None = None
    rule: MaskRules = NotImplemented
    # I think this is not actually needed

    def create_mask_key(self, value: list[list[str]], current: str) -> MaskKey:
        rows = ["".join(row).replace(" ", "_") for row in value]
        return MaskKey(
            mask_type=self.__class__,
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


@dataclass
class MaskHorizontal3Centered(AbstractMask):
    rule: MaskRules = MaskRules(
        rows_above=0, rows_below=0, columns_left=1, columns_right=1
    )


@dataclass
class MaskHorizontal3CenteredX(MaskHorizontal3Centered):
    match_symbol: str | None = "X"


@dataclass
class MaskHorizontal3CenteredO(MaskHorizontal3Centered):
    match_symbol: str | None = "O"


@dataclass
class MaskHorizontal3Left(AbstractMask):
    rule: MaskRules = MaskRules(
        rows_above=0, rows_below=0, columns_left=0, columns_right=2
    )


@dataclass
class MaskHorizontal3LeftX(MaskHorizontal3Left):
    match_symbol: str | None = "X"


@dataclass
class MaskHorizontal3LeftO(MaskHorizontal3Left):
    match_symbol: str | None = "O"


@dataclass
class MaskHorizontal3Right(AbstractMask):
    rule: MaskRules = MaskRules(
        rows_above=0, rows_below=0, columns_left=2, columns_right=0
    )


@dataclass
class MaskHorizontal3RightX(MaskHorizontal3Right):
    match_symbol: str | None = "X"


@dataclass
class MaskHorizontal3RightO(MaskHorizontal3Right):
    match_symbol: str | None = "O"


@dataclass
class MaskVertical3Centered(AbstractMask):
    rule: MaskRules = MaskRules(
        rows_above=1, rows_below=1, columns_left=0, columns_right=0
    )


@dataclass
class MaskVertical3CenteredX(MaskVertical3Centered):
    match_symbol: str | None = "X"


@dataclass
class MaskVertical3CenteredO(MaskVertical3Centered):
    match_symbol: str | None = "O"


@dataclass
class MaskVertical3Above(AbstractMask):
    rule: MaskRules = MaskRules(
        rows_above=0, rows_below=2, columns_left=0, columns_right=0
    )


@dataclass
class MaskVertical3AboveX(MaskVertical3Above):
    match_symbol: str | None = "X"


@dataclass
class MaskVertical3AboveO(MaskVertical3Above):
    match_symbol: str | None = "O"


@dataclass
class MaskVertical3Below(AbstractMask):
    rule: MaskRules = MaskRules(
        rows_above=2, rows_below=0, columns_left=0, columns_right=0
    )


@dataclass
class MaskVertical3BelowX(MaskVertical3Below):
    match_symbol: str | None = "X"


@dataclass
class MaskVertical3BelowO(MaskVertical3Below):
    match_symbol: str | None = "O"


ALL_MASKS: list[AbstractMask] = [
    MaskHorizontal3Centered(),
    MaskHorizontal3CenteredX(),
    MaskHorizontal3CenteredO(),
    MaskHorizontal3Left(),
    MaskHorizontal3LeftX(),
    MaskHorizontal3LeftO(),
    MaskHorizontal3Right(),
    MaskHorizontal3RightX(),
    MaskHorizontal3RightO(),
    MaskVertical3Centered(),
    MaskVertical3CenteredX(),
    MaskVertical3CenteredO(),
    MaskVertical3Above(),
    MaskVertical3AboveX(),
    MaskVertical3AboveO(),
    MaskVertical3Below(),
    MaskVertical3BelowX(),
    MaskVertical3BelowO(),
]
