import random
from typing import Any

from gridworld.constants import SIMPLE_ACTIONS


class RandomAgent:
    def __init__(self, *, rng: random.Random | None = None):
        """
        rng is for testing purposes, to seed the random number generator.
        """
        self.rng = rng or random.Random()

    def act(self, state: tuple[int, int]) -> str:
        return self.rng.choice(SIMPLE_ACTIONS)

    def observe(self, *args: Any, **kwargs: Any):
        pass

    def reset(self, **kwargs: Any):
        pass


if __name__ == "__main__":
    assert SIMPLE_ACTIONS, "No actions defined!"
    agent = RandomAgent()
    for i in range(10):
        action = agent.act((0, 0))
        print(f"Action {i}: {action}")
