from dataclasses import dataclass


X = "X"
O = "O"
E = " "


@dataclass
class StepResult:
    coordinate: tuple[int, int]
    score: int
    symbol: str
    loss_reason: str = ""
