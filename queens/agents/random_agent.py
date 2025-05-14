from typing import Any

from queens.agents.generic_agent import Agent
from queens.dtos import Observation, StepResult, RunnerReturn
from numpy.typing import NDArray
import numpy as np


class RandomAgent(Agent):
    def _select_row(self, board: NDArray[np.int_]) -> int:
        return self.rng.randint(0, board.shape[0] - 1)

    def _select_column(self, board: NDArray[np.int_]) -> int:
        return self.rng.randint(0, board.shape[0] - 1)

    def act(self, observation: Observation) -> tuple[int, int]:
        row = self._select_row(observation.board_state.board)
        col = self._select_column(observation.board_state.board)
        return row, col

    def observe_step(self, result: StepResult):
        pass

    def reset(self, **kwargs: Any):
        pass

    def observe_result(self, result: RunnerReturn):
        pass


if __name__ == "__main__":
    agent = RandomAgent()
