from abc import ABC, abstractmethod
from typing import Any
from tic_tac_logic.constants import StepResult, Observation


class Agent(ABC):
    explain: bool = False

    def __init__(self, rows: int, columns: int) -> None:
        """Common agent initialisation.

        Parameters
        ----------
        rows:
            Number of rows the agent will operate on.
        columns:
            Number of columns the agent will operate on.
        """
        self.rows = rows
        self.columns = columns
        self._q_table: dict[Any, Any] = {}

    @abstractmethod
    def act(self, observation: Observation) -> tuple[tuple[int, int], str]:
        """
        Decide on an action based on the current grid and the agent's symbol.

        :param grid: The current state of the grid.
        :param symbol: The symbol of the agent (X or O).
        :return: A tuple representing the chosen action (row, column).
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Reset the agent's state.
        This is useful for agents that maintain some internal state across episodes.
        """
        pass

    @abstractmethod
    def learn(self, step_result: StepResult):
        """
        Update the agent's knowledge based on the action taken and the reward received.

        :param state: The previous state of the grid.
        :param action: The action taken (row, column).
        :param reward: The reward received after taking the action.
        :param next_state: The new state of the grid after the action.
        """
        pass

    @property
    def q_table(self) -> dict[Any, Any]:
        return self._q_table

    @q_table.setter
    def q_table(self, value: dict[Any, Any]) -> None:
        self._q_table = value
