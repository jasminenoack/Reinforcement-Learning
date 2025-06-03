from collections import defaultdict
import random
from tic_tac_logic.constants import PLACEMENT_OPTIONS, StepResult, Observation, E
from tic_tac_logic.agents.base_agent import Agent


class RLShapingBasedAgent(Agent):
    def __init__(self, grid: list[list[str]]) -> None:
        self.q_table: dict[tuple[int, int], dict[str, float]] = {}
        self.rows = len(grid)
        self.columns = len(grid[0])
        for row in range(self.rows):
            for column in range(self.columns):
                self.q_table[(row, column)] = {
                    option: 0.0 for option in PLACEMENT_OPTIONS
                }

        self.placement_options = PLACEMENT_OPTIONS
        self.full_discovery = 0.1
        self.mild_discovery = 0.3
        self.learning_rate = 0.1
        self.epsilon_decay = 0.999
        self.reset()

    def reset(self) -> None:
        for coordinate in self.q_table:
            for action in self.q_table[coordinate]:
                self.q_table[coordinate][action] = min(
                    max(0, self.q_table[coordinate][action]), 20
                )

    def has_neighboring_actions(
        self, coordinate: tuple[int, int], observation: Observation
    ) -> bool:
        row, column = coordinate
        orthogonal_neighbors = [
            (row - 1, column),
            (row + 1, column),
            (row, column - 1),
            (row, column + 1),
        ]
        orthogonal_neighbors = [
            (i, j)
            for i, j in orthogonal_neighbors
            if 0 <= i < self.rows and 0 <= j < self.columns
        ]
        for neighbor in orthogonal_neighbors:
            if observation.grid[neighbor[0]][neighbor[1]] in self.placement_options:
                return True

        return False

    def _get_actions_by_value(
        self, observation: Observation
    ) -> dict[float, list[tuple[tuple[int, int], str]]]:
        action_by_values: dict[float, list[tuple[tuple[int, int], str]]] = defaultdict(
            list
        )
        for coordinate, actions in self.q_table.items():
            for action, value in actions.items():
                if observation.grid[coordinate[0]][
                    coordinate[1]
                ] == E and self.has_neighboring_actions(coordinate, observation):
                    action_by_values[value].append((coordinate, action))
        return action_by_values

    def get_top_actions(
        self,
        action_count: int | float,
        action_by_values: dict[float, list[tuple[tuple[int, int], str]]],
    ):
        """
        This returns the smallest number of actions that is larger than the count of actions we want to look at.
        """
        top_actions: list[tuple[tuple[int, int], str]] = []
        sorted_values = sorted(action_by_values.keys(), reverse=True)
        for value in sorted_values:
            if len(top_actions) >= action_count:
                break
            actions = action_by_values[value]
            top_actions.extend(actions)
        return top_actions

    def act(self, observation: Observation) -> tuple[tuple[int, int], str]:
        discovery_rate = random.random()
        if self.full_discovery > discovery_rate:
            number_of_actions = float("inf")
        elif self.mild_discovery > discovery_rate:
            number_of_actions = 5
        else:
            number_of_actions = 3
        actions = self._get_actions_by_value(observation)
        top_actions = self.get_top_actions(number_of_actions, actions)
        action = random.choice(top_actions)
        return action

    def learn(
        self,
        step_result: StepResult,
    ):
        coordinate = step_result.coordinate
        symbol = step_result.symbol
        score = step_result.score
        self.q_table[coordinate][symbol] += score * self.learning_rate
        # self.q_table[coordinate][symbol] += (1 - self.learning_rate) * self.q_table[
        #     coordinate
        # ][symbol] + self.learning_rate * score
        self.full_discovery *= self.epsilon_decay
        self.mild_discovery *= self.epsilon_decay
