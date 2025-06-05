from collections import defaultdict
from dataclasses import dataclass
import random
from typing import TypedDict
from tic_tac_logic.agents.base_agent import Agent
from tic_tac_logic.constants import StepResult, E, PLACEMENT_OPTIONS, Observation
import logging

logging.basicConfig(
    filename="tic_tac_logic/mask_agent.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MaskKey:
    mask_type: type["AbstractMask"]
    pattern: str
    symbol: str
    placement_location: int


@dataclass
class AbstractMask:
    symbol: str = NotImplemented
    # I think this is not actually needed
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


@dataclass(frozen=True)
class ConfidentMask:
    mask_key: MaskKey
    # prediction is from  -1 for failure, 1 for success
    prediction: float


@dataclass
class QTable(TypedDict):
    masks: dict[MaskKey, MaskResult]


class MaskAgent(Agent):
    confidence_threshold = 5

    def __init__(self, grid: list[list[str]]) -> None:
        self.q_table: QTable = {  # pyright: ignore[reportIncompatibleVariableOverride]
            "masks": {}
        }
        self.rows = len(grid)
        self.columns = len(grid[0])
        self.epsilon = 0.1
        self.masks = [
            MaskHorizontal3(),
            MaskHorizontal3X(),
            MaskHorizontal3O(),
        ]

    def log(self, message: str) -> None:
        if self.explain:
            logger.info(message)

    def empty_cells(self, grid: list[list[str]]) -> list[tuple[int, int]]:
        return [
            (row_i, col_i)
            for row_i in range(self.rows)
            for col_i in range(self.columns)
            if grid[row_i][col_i] == E
        ]

    def find_aggressive_failures(self) -> set[ConfidentMask]:
        """
        The goal here is to find cells that are particularly bad for the agent to place in.
        Basically, for us to decide that a mask is objectively bad we actually want to see that
        it is a consistent failure, and that it has failed enough times for us to be confident it
        is a bad choice.

        There are some issues here if you use a single puzzle there could be a confounding factor.
        """
        q_table = self.q_table["masks"]
        failures: set[ConfidentMask] = set()
        for mask_key, mask_result in q_table.items():
            if mask_result.failure_count < 5:
                continue
            if (
                mask_result.success_count == 0
                and mask_result.failure_count >= self.confidence_threshold
            ):
                failures.add(
                    ConfidentMask(
                        mask_key=mask_key,
                        prediction=-1.0,  # -1 for failure
                    )
                )
        return failures

    def remove_failing_options(
        self,
        grid: list[list[str]],
        possible_moves: set[tuple[tuple[int, int], str]],
        failure_masks: set[ConfidentMask],
    ):
        bad_moves: set[tuple[tuple[int, int], str]] = set()
        for move in possible_moves:
            cell, symbol = move
            for mask in self.masks:
                mask = mask.get_mask(cell, grid, current=symbol)
                if (
                    mask
                    and ConfidentMask(mask_key=mask, prediction=-1) in failure_masks
                ):
                    bad_moves.add((cell, symbol))
                    break

        return possible_moves - bad_moves

    def options_with_one_choice(
        self, possible_moves: set[tuple[tuple[int, int], str]]
    ) -> set[tuple[tuple[int, int], str]]:
        by_coordinate: dict[tuple[int, int], set[str]] = defaultdict(set)
        for cell, symbol in possible_moves:
            by_coordinate[cell].add(symbol)

        return {
            (cell, symbols.pop())
            for cell, symbols in by_coordinate.items()
            if len(symbols) == 1
        }

    def do_not_discover(self):
        return random.random() > self.epsilon

    def act(self, observation: Observation) -> tuple[tuple[int, int], str]:
        self.log("Acting based on observation")
        empty_cells = self.empty_cells(observation.grid)
        if not empty_cells:
            raise ValueError("No empty cells available for placement.")

        possible_moves = set(
            (cell, symbol) for cell in empty_cells for symbol in PLACEMENT_OPTIONS
        )

        if self.do_not_discover():
            self.log("    Removing known issues")
            # remove known bad placements
            failure_masks = self.find_aggressive_failures()
            possible_moves = self.remove_failing_options(
                observation.grid, possible_moves, failure_masks
            )

        if self.do_not_discover():
            self.log("    Looking for best choices")
            best_options = self.options_with_one_choice(possible_moves)
            if best_options:
                result = best_options.pop()
                self.log(
                    f"        Returning best option {result[0]} with symbol {result[1]}"
                )
                return result

        random_cell = random.choice(empty_cells)
        random_symbol = random.choice(PLACEMENT_OPTIONS)
        self.log(f"    Returning random cell {random_cell} with symbol {random_symbol}")
        return random_cell, random_symbol

    def learn(self, step_result: StepResult) -> None:
        masks = self.masks
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
        self.log("Starting a new episode, resetting agent state.")
        self.current_grid = None
