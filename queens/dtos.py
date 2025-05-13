from dataclasses import dataclass, field
from numpy.typing import NDArray
import numpy as np


@dataclass
class Result:
    action: tuple[int, int]


@dataclass
class BoardState:
    pass


@dataclass
class Observation:
    board_state: BoardState = field(default_factory=BoardState)


@dataclass
class RunnerReturn:
    trajectory: list[Result]
    solved: bool
    board: NDArray[np.int_]
    moves: int
    score: int
