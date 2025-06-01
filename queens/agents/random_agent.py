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


class RandomAgentByRow(RandomAgent):
    def reset(self, **kwargs: Any):
        super().reset(**kwargs)
        self.current_row = 0

    def _select_row(self, board: NDArray[np.int_]) -> int:
        row = self.current_row
        self.current_row += 1
        if self.current_row > board.shape[0]:
            raise ValueError("No more rows available to select.")
        return row


class RandomAgentAlsoByColumn(RandomAgentByRow):
    def reset(self, **kwargs: Any):
        super().reset(**kwargs)
        self.current_column = 0

    def _select_column(self, board: NDArray[np.int_]) -> int:
        used_columns = set(int(x) for x in np.where(board == 1)[1])
        all_columns = set(range(board.shape[1]))
        remaining_columns = all_columns - used_columns
        column = self.rng.choice(list(remaining_columns))
        return column


if __name__ == "__main__":
    agent = RandomAgent()
