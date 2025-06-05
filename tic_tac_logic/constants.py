from dataclasses import dataclass
from typing import Any


X = "X"
O = "O"
E = " "

PLACEMENT_OPTIONS = [X, O]


@dataclass
class StepResult:
    coordinate: tuple[int, int]
    score: int
    symbol: str
    loss_reason: str = ""
    pre_step_grid: list[list[str]] | None = None
    grid: list[list[str]] | None = None


@dataclass
class Result:
    actions: int
    score: int
    won: bool
    q_table: dict[tuple[int, int], dict[str, float]] | dict[str, Any] | Any | None = (
        None
    )
    error: str | None = None


@dataclass
class Observation:
    grid: list[list[str]]
