from dataclasses import dataclass
from enum import Enum
import random
from tic_tac_logic.agents.base_agent import Agent
from tic_tac_logic.constants import StepResult, E, PLACEMENT_OPTIONS, Observation

FAILURE = "FAILURE"


class FailureClass(Enum):
    ROW = "ROW"
    COLUMN = "COLUMN"
    BOARD = "BOARD"


# masks?

"""
So here were my thoughts. I was thinking we could get it sort of POC working. Then we could thing about a few things:

1. back porting failure (if there are no remaining valid states your current state is also a failure)
2. adding more validation (like column)
3. trying to generate more like "masking" than just checking entire areas to lower the memory space
"""


@dataclass(frozen=True)
class Move:
    type: FailureClass
    applicable_area: str
    location: tuple[int, int] | int
    symbol: str

    @classmethod
    def create(
        cls,
        type: FailureClass,
        applicable_area: str | list[str] | list[list[str]],
        location: tuple[int, int] | int,
        symbol: str,
    ) -> "Move":
        applicable_area_str = ""
        if isinstance(applicable_area, str):
            applicable_area_str = applicable_area
        else:
            for item in applicable_area:
                if isinstance(item, str):
                    applicable_area_str += item + ", "
                else:
                    applicable_area_str += ", ".join(item) + ",\n"
        return cls(
            type=type,
            applicable_area=applicable_area_str,
            location=location,
            symbol=symbol,
        )


@dataclass(frozen=True)
class ValidPlacement:
    coordinate: tuple[int, int]
    symbol: str


class FailureAgent(Agent):
    def __init__(self, rows: int, columns: int) -> None:
        super().__init__(rows, columns)
        self.q_table: dict[str, set[Move]] = {"failures": set()}

    def _row_move(
        self, grid: list[list[str]], coordinate: tuple[int, int], player: str
    ) -> Move:
        row_index, col_index = coordinate
        row = grid[row_index]
        return Move.create(
            type=FailureClass.ROW,
            applicable_area=row,
            location=col_index,
            symbol=player,
        )

    def _column_move(
        self, grid: list[list[str]], coordinate: tuple[int, int], player: str
    ) -> Move:
        col_index = coordinate[1]
        column = [grid[row_index][col_index] for row_index in range(len(grid))]
        return Move.create(
            type=FailureClass.COLUMN,
            applicable_area=column,
            location=coordinate[0],
            symbol=player,
        )

    def _board_move(
        self, grid: list[list[str]], coordinate: tuple[int, int], player: str
    ) -> Move:
        return Move.create(
            type=FailureClass.BOARD,
            applicable_area=grid,
            location=coordinate,
            symbol=player,
        )

    def get_valid_placements(self, grid: list[list[str]]) -> set[ValidPlacement]:
        valid_placements: set[ValidPlacement] = set()
        for row_index in range(self.rows):
            for col_index in range(self.columns):
                if grid[row_index][col_index] == E:
                    for player in PLACEMENT_OPTIONS:
                        moves_to_check = set(
                            [
                                self._row_move(grid, (row_index, col_index), player),
                                self._column_move(grid, (row_index, col_index), player),
                                self._board_move(grid, (row_index, col_index), player),
                            ]
                        )
                        remaining_moves = moves_to_check - self.q_table["failures"]
                        if len(moves_to_check) == len(remaining_moves):
                            valid_placements.add(
                                ValidPlacement(
                                    coordinate=(row_index, col_index),
                                    symbol=player,
                                )
                            )
        return valid_placements

    def act(self, observation: Observation) -> tuple[tuple[int, int], str]:
        valid_placements = self.get_valid_placements(observation.grid)
        if valid_placements:
            placement = random.choice(list(valid_placements))
            return (
                placement.coordinate,
                placement.symbol,
            )

        raise ValueError("No valid placements")

    def learn(self, step_result: StepResult) -> None:
        q_table = self.q_table
        lost = step_result.loss_reason
        if lost:
            if "ROW" in lost:
                assert step_result.pre_step_grid
                move = self._row_move(
                    step_result.pre_step_grid,
                    step_result.coordinate,
                    step_result.symbol,
                )
                q_table["failures"].add(move)
            elif "COLUMN" in lost:
                assert step_result.pre_step_grid
                move = self._column_move(
                    step_result.pre_step_grid,
                    step_result.coordinate,
                    step_result.symbol,
                )
                q_table["failures"].add(move)
            else:
                assert step_result.pre_step_grid
                move = self._board_move(
                    step_result.pre_step_grid,
                    step_result.coordinate,
                    step_result.symbol,
                )
                q_table["failures"].add(move)
        else:
            # if we know there are no remaining valid placments
            assert step_result.pre_step_grid
            assert step_result.grid
            next_valid_placements = self.get_valid_placements(step_result.grid)
            if not next_valid_placements:
                move = Move.create(
                    type=FailureClass.BOARD,
                    applicable_area=step_result.pre_step_grid,
                    location=step_result.coordinate,
                    symbol=step_result.symbol,
                )
                q_table["failures"].add(move)

    def reset(self) -> None:
        # Reset the agent's state
        ...
