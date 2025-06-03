from dataclasses import dataclass
from enum import Enum
from tic_tac_logic.agents.base_agent import Agent
from tic_tac_logic.constants import StepResult, E, PLACEMENT_OPTIONS, Observation

FAILURE = "FAILURE"


class FailureClass(Enum):
    ROW = "ROW"
    COLUMN = "COLUMN"


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
            applicable_area=applicable_area_str.strip(),
            location=location,
            symbol=symbol,
        )


@dataclass(frozen=True)
class ValidPlacement:
    coordinate: tuple[int, int]
    symbol: str


class FailureAgent(Agent):
    def __init__(self, grid: list[list[str]]) -> None:
        self.q_table: dict[str, set[Move]] = {"failures": set()}
        self.rows = len(grid)
        self.columns = len(grid[0])

    def save_failure(self, move: tuple[int, int], player: str) -> None: ...

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

    def get_valid_placements(self, grid: list[list[str]]) -> set[ValidPlacement]:
        valid_placements: set[ValidPlacement] = set()
        for row_index in range(self.rows):
            for col_index in range(self.columns):
                if grid[row_index][col_index] == E:
                    for player in PLACEMENT_OPTIONS:
                        moves_to_check = set(
                            [self._row_move(grid, (row_index, col_index), player)]
                        )
                        if not moves_to_check & self.q_table["failures"]:
                            valid_placements.add(
                                ValidPlacement(
                                    coordinate=(row_index, col_index),
                                    symbol=player,
                                )
                            )
                        else:
                            print("BAD MOVE", row_index, col_index, player)
        return valid_placements

    def act(self, observation: Observation) -> tuple[tuple[int, int], str]:
        valid_placements = self.get_valid_placements(observation.grid)
        if not valid_placements:
            raise ValueError("No valid placements available.")
        # TODO, make this way less hardcoded
        # for testing purposes we are looking at the exact movements
        moves_interesting_moves = sorted(
            [
                placement
                for placement in valid_placements
                if placement.coordinate in [(1, 2), (1, 3)]
            ],
            key=lambda x: (x.coordinate, x.symbol),
        )
        if moves_interesting_moves:
            return (
                moves_interesting_moves[0].coordinate,
                moves_interesting_moves[0].symbol,
            )

        if valid_placements:
            placement = valid_placements.pop()
            return (
                placement.coordinate,
                placement.symbol,
            )

        raise ValueError("I dunno")

    def learn(self, step_result: StepResult) -> None:
        q_table = self.q_table
        lost = step_result.loss_reason
        if lost:
            if "ROW" in lost:
                assert step_result.pre_step_grid
                row = step_result.pre_step_grid[step_result.coordinate[0]]
                move = Move.create(
                    type=FailureClass.ROW,
                    applicable_area=row,
                    location=step_result.coordinate[1],
                    symbol=step_result.symbol,
                )
                q_table["failures"].add(move)

    def reset(self) -> None:
        # Reset the agent's state
        ...
