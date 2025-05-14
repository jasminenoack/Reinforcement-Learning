from dataclasses import dataclass, field
from enum import Enum
from numpy.typing import NDArray
import numpy as np


class FailureType(Enum):
    ROW = "row"
    COLUMN = "column"
    DIAGONAL = "diagonal"
    REVERSE_DIAGONAL = "reverse_diagonal"


@dataclass
class BoardState:
    board: NDArray[np.int_] = NotImplemented


@dataclass
class StepResult:
    action: tuple[int, int]
    reward: int = NotImplemented
    failure_type: FailureType | None = NotImplemented
    board_state: BoardState = field(default_factory=BoardState)


@dataclass
class Observation:
    board_state: BoardState = field(default_factory=BoardState)


@dataclass
class RunnerReturn:
    trajectory: list[StepResult]
    solved: bool
    board: NDArray[np.int_]
    moves: int
    score: int
