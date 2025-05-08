from abc import ABC, abstractmethod
from typing import Any


class Agent(ABC):
    def __init__(
        self,
        rows: int = 5,
        cols: int = 5,
        goal: tuple[int, int] = (4, 4),
    ):
        self.rows = rows
        self.cols = cols
        self.goal = goal
        self.reset()

    @abstractmethod
    def act(self, *args: Any, **kwargs: Any) -> str: ...
    @abstractmethod
    def observe(self, *args: Any, **kwargs: Any): ...
    @abstractmethod
    def reset(self, **kwargs: Any): ...
