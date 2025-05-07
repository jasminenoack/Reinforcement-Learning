from abc import ABC, abstractmethod
from typing import Any


class Agent(ABC):
    @abstractmethod
    def act(self, *args: Any, **kwargs: Any) -> str: ...
    @abstractmethod
    def observe(self, *args: Any, **kwargs: Any): ...
    @abstractmethod
    def reset(self, **kwargs: Any): ...
