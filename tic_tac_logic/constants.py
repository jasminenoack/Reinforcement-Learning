from dataclasses import dataclass


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


@dataclass
class Result:
    actions: int
    score: int
    won: bool
    q_table: dict[tuple[int, int], dict[str, float]] | None = None


@dataclass
class Observation:
    grid: list[list[str]]
