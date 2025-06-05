from dataclasses import dataclass
from dataclasses import dataclass
import random
from typing import TypedDict
from tic_tac_logic.agents.base_agent import Agent
from tic_tac_logic.constants import StepResult, E, PLACEMENT_OPTIONS, Observation


@dataclass(frozen=True)
class MaskKey:
    mask_type: type["AbstractMask"]
    pattern: str
    symbol: str
    placement_location: int


@dataclass
class AbstractMask:
    symbol: str = NotImplemented
    placement_location: int = 1

    def get_mask(self, coord: tuple[int, int], grid: list[list[str]]) -> str:
        raise NotImplementedError()

    def create_mask_key(self, value: list[str], current: str) -> MaskKey:
        return MaskKey(
            mask_type=self.__class__,
            pattern="".join(value).replace(" ", "_"),
            symbol=current,
            placement_location=self.placement_location,
        )

    def remove_non_matching(self, grid: list[list[str]]) -> list[list[str]]:
        return [[cell if cell == self.symbol else E for cell in row] for row in grid]


@dataclass
class MaskHorizontal3(AbstractMask):
    def get_mask(
        self, coord: tuple[int, int], grid: list[list[str]], current: str
    ) -> MaskKey | None:
        row_i, col_i = coord
        row = grid[row_i]
        row_length = len(row)
        first_item = col_i == 0
        last_item = col_i == row_length - 1
        if first_item or last_item:
            return None
        return self.create_mask_key(row[col_i - 1 : col_i + 2], current=current)


@dataclass
class MaskHorizontal3X(MaskHorizontal3):
    symbol: str = "X"

    def get_mask(
        self, coord: tuple[int, int], grid: list[list[str]], current: str
    ) -> MaskKey | None:
        grid = self.remove_non_matching(grid)
        return super().get_mask(coord, grid, current=current)


@dataclass
class MaskHorizontal3O(MaskHorizontal3):
    symbol: str = "O"
    failure_mode_requirement = 5

    def get_mask(
        self, coord: tuple[int, int], grid: list[list[str]], current: str
    ) -> MaskKey | None:
        grid = self.remove_non_matching(grid)
        return super().get_mask(coord, grid, current=current)


@dataclass
class MaskResult:
    mask: MaskKey
    failure_count: int
    success_count: int


@dataclass
class QTable(TypedDict):
    masks: dict[MaskKey, MaskResult]


class MaskAgent(Agent):
    def __init__(self, grid: list[list[str]]) -> None:
        self.q_table: QTable = {  # pyright: ignore[reportIncompatibleVariableOverride]
            "masks": {}
        }
        self.rows = len(grid)
        self.columns = len(grid[0])

    def empty_cells(self, grid: list[list[str]]) -> list[tuple[int, int]]:
        return [
            (row_i, col_i)
            for row_i in range(self.rows)
            for col_i in range(self.columns)
            if grid[row_i][col_i] == E
        ]

    def find_agressive_failures(self) -> tuple[tuple[int, int], str]:
        """
        The goal here is to find cells that are particularly bad for the agent to place in.
        Basically, for us to decide that a mask is objectively bad we actually want to see that
        it is a consistent failure, and that it has failed enough times for us to be confident it
        is a bad choice.

        There are some issues here if you use a single puzzle there could be a confounding factor.
        """

    def act(self, observation: Observation) -> tuple[tuple[int, int], str]:
        empty_cells = self.empty_cells(observation.grid)
        if not empty_cells:
            raise ValueError("No empty cells available for placement.")
        random_cell = random.choice(empty_cells)
        random_symbol = random.choice(PLACEMENT_OPTIONS)
        return random_cell, random_symbol

    def learn(self, step_result: StepResult) -> None:
        masks = [
            MaskHorizontal3(),
            MaskHorizontal3X(),
            MaskHorizontal3O(),
        ]
        assert step_result.pre_step_grid
        masks = [
            mask.get_mask(
                step_result.coordinate, step_result.pre_step_grid, step_result.symbol
            )
            for mask in masks
        ]
        masks = set([mask for mask in masks if mask is not None])
        lost = step_result.loss_reason
        q_table = self.q_table["masks"]
        for mask in masks:
            if mask not in q_table:
                q_table[mask] = MaskResult(
                    mask=mask,
                    failure_count=0,
                    success_count=0,
                )
            if lost:
                q_table[mask].failure_count += 1
            else:
                q_table[mask].success_count += 1

    def reset(self) -> None:
        pass
