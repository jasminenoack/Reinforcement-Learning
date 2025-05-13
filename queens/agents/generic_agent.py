from abc import ABC, abstractmethod
import random
from typing import Any

from queens.dtos import Observation, Result


class Agent(ABC):
    def __init__(self, *, rng: random.Random | None = None, **kwargs: Any):
        """
        This is rather like the random agent, but it strongly prefers to not repeat a move
        """
        self.rng = rng or random.Random()
        super().__init__(**kwargs)

    @abstractmethod
    def act(self, observation: Observation) -> tuple[int, int]:
        pass

    @abstractmethod
    def observe(self, result: Result):
        pass

    @abstractmethod
    def reset(self, **kwargs: Any):
        pass
