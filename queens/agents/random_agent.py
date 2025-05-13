from typing import Any

from queens.agents.generic_agent import Agent
from queens.dtos import Observation, Result


class RandomAgent(Agent):
    def act(self, observation: Observation) -> tuple[int, int]:
        row = self.rng.randint(0, 7)
        col = self.rng.randint(0, 7)
        return row, col

    def observe(self, result: Result):
        pass

    def reset(self, **kwargs: Any):
        pass


if __name__ == "__main__":
    agent = RandomAgent()
