"""
GridworldEnv: A simple, extensible environment for reinforcement learning agents.

This environment is built to support gradual exploration of RL concepts, beginning with
simple open grids and building toward more complex, constrained environments.

Core Responsibilities:
- Set up a 2D grid and place the agent and goal.
- Accept actions (up, down, left, right).
- Check for valid movement and update the agent's position.
- Return:
    - The new state
    - A reward
    - A done flag (True if the goal is reached)

This class follows the familiar step()/reset() structure used in OpenAI Gym environments.

Phases of Complexity:
1. Basic Gridworld — Open Space:
   - No obstacles or traps.
   - The agent learns the shortest path from start to goal.
   - Useful for visualizing basic learning behavior over time.

2. Gridworld with Obstacles:
   - Add walls or negative-reward tiles.
   - The agent must learn to navigate around hazards.

3. Randomized Grids / Curriculum Learning:
   - Train on varied grid configurations.
   - Test agent generalization to unseen layouts.

4. Hard Mode:
   - Narrow corridors, decoy tiles, and delayed rewards.
   - Moves toward the complexity of logic puzzles like Sudoku.
   - Complex penalties

Future extensions may include visualization tools (e.g., heatmaps, path tracking),
Q-table inspection, and dynamic difficulty.

Example usage:
    env = GridworldEnv(rows=5, cols=5)
    env.reset()
    env.render()
"""

from collections import defaultdict
from dataclasses import dataclass
from unittest.mock import ANY
from matplotlib.pylab import dirichlet
from rich.console import Console
from rich.text import Text

from gridworld.utils import (
    DOWN,
    GOAL,
    LEFT,
    MOVEMENT,
    OFF_BOARD,
    RIGHT,
    UP,
    WALL,
    render_heatmap,
)

console = Console()


@dataclass
class ActionConfig:
    row_movement: int
    col_movement: int


ACTIONS = {
    UP: ActionConfig(-1, 0),
    DOWN: ActionConfig(1, 0),
    LEFT: ActionConfig(0, -1),
    RIGHT: ActionConfig(0, 1),
}

STEP_RESULT_REWARD = {
    OFF_BOARD: -10,
    WALL: -10,
    GOAL: 100,
    MOVEMENT: -1,
}


@dataclass
class StepResult:
    new_state: tuple[int, int]
    reward: float
    done: bool

    def __iter__(self):
        yield self.new_state
        yield self.reward
        yield self.done


class VisitCounter:
    def __init__(self, data: dict = None) -> None:
        self.data = data or defaultdict(int)

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __eq__(self, value: "dict | VisitCounter") -> bool:
        if isinstance(value, dict):
            value_comparison = {
                key: value for key, value in value.items() if value != 0
            }
            data_comparison = {
                key: value for key, value in self.data.items() if value != 0
            }
            return value_comparison == data_comparison
        if isinstance(value, VisitCounter):
            value_comparison = {
                key: value for key, value in value.data.items() if value != 0
            }
            data_comparison = {
                key: value for key, value in self.data.items() if value != 0
            }

            return data_comparison == value_comparison
        if value is ANY:
            return True
        return False

    def __add__(self, other: "VisitCounter") -> "VisitCounter":
        for coordinate in other.data.keys():
            self[coordinate] += other[coordinate]
        return self

    @classmethod
    def avg(cls, *visit_counts: list["VisitCounter"]) -> "VisitCounter":
        coordinates = set()
        total_counters = len(visit_counts)
        new_counter = cls()
        # get all the coordinates
        for visit_count in visit_counts:
            coordinates.update(visit_count.data.keys())
        # create avg counter
        for coordinate in coordinates:
            total = sum(visit_count.data[coordinate] for visit_count in visit_counts)
            avg = total / total_counters
            new_counter[coordinate] = avg
        return new_counter

    def items(self):
        return self.data.items()

    def values(self):
        return self.data.values()


@dataclass
class Cell:
    agent: bool = False
    goal: bool = False
    _wall: bool = False
    visited: int = 0

    @property
    def wall(self) -> bool:
        return self._wall

    @wall.setter
    def wall(self, value: bool) -> None:
        if self.agent:
            raise ValueError("Cannot set wall on a cell with an agent.")
        if self.goal:
            raise ValueError("Cannot set wall on a cell with a goal.")
        self._wall = value


class GridWorldEnv:
    def __init__(
        self,
        *,
        max_steps: int = 100,
        rows: int = 5,
        cols: int = 5,
        walls: list[tuple] | None = None,
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.start = (0, 0)
        self.goal = (rows - 1, cols - 1)
        self.reward_config = STEP_RESULT_REWARD
        self.max_steps = max_steps
        self._setup()
        for coordinate in walls or []:
            self.get_cell(coordinate).wall = True

    def _setup(self) -> None:
        self.total_reward = 0
        self.current_step = 0
        self.grid = [[Cell() for _ in range(self.cols)] for _ in range(self.rows)]
        self.get_cell(self.start).agent = True
        self.get_cell(self.goal).goal = True
        self.visit(new_pos=self.agent_pos)

    @property
    def visit_counts(self) -> VisitCounter:
        return VisitCounter(
            data={
                (i, j): self.grid[i][j].visited
                for i in range(self.rows)
                for j in range(self.cols)
            }
        )

    @property
    def reached_goal(self) -> bool:
        goal = self.find_goal_position()
        goal_cell = self.get_cell(goal)
        return goal_cell.visited > 0 and goal_cell.agent

    @property
    def done(self) -> bool:
        return self.reached_goal or self.current_step >= self.max_steps

    @property
    def agent_pos(self) -> tuple[int, int]:
        return self.find_agent_position()

    @agent_pos.setter
    def agent_pos(self, pos: tuple[int, int]) -> None:
        current_agent_pos = self.find_agent_position()
        self.get_cell(current_agent_pos).agent = False
        self.get_cell(pos).agent = True
        self.visit(new_pos=pos)

    def get_cell(self, pos: tuple[int, int]) -> Cell:
        if not (0 <= pos[0] < self.rows and 0 <= pos[1] < self.cols):
            raise ValueError(f"Position {pos} is out of bounds.")
        return self.grid[pos[0]][pos[1]]

    def find_agent_position(self) -> tuple[int, int]:
        for i in range(self.rows):
            for j in range(self.cols):
                if self.grid[i][j].agent:
                    return (i, j)
        raise ValueError("Agent not found in the grid.")

    def find_goal_position(self) -> tuple[int, int]:
        for i in range(self.rows):
            for j in range(self.cols):
                if self.grid[i][j].goal:
                    return (i, j)
        raise ValueError("Goal not found in the grid.")

    def reset(self) -> None:
        self._setup()
        return self.agent_pos

    def render(self) -> None:
        console.print(f"Step: {self.current_step} of {self.max_steps}")
        console.print(f"Total Reward: {round(self.total_reward, 2)}")
        console.print()
        agent_pos = self.agent_pos
        goal_pos = self.goal
        for i in range(self.rows):
            line = Text()
            for j in range(self.cols):
                if (i, j) == agent_pos:
                    line.append("A ", style="bold blue")
                elif (i, j) == goal_pos:
                    line.append("G ", style="bold green")
                else:
                    line.append(". ", style="white")
            console.print(line)

    def get_state(self) -> tuple[int, int]:
        return self.agent_pos

    def visit(self, new_pos: tuple[int, int]) -> None:
        self.grid[new_pos[0]][new_pos[1]].visited += 1

    def next_cell(self, action: str) -> tuple[tuple[int, int], str]:
        pos = self.agent_pos
        action_config = ACTIONS[action]
        new_row = pos[0] + action_config.row_movement
        new_col = pos[1] + action_config.col_movement
        if not (0 <= new_row < self.rows and 0 <= new_col < self.cols):
            return (pos, OFF_BOARD)

        new_cell = self.get_cell((new_row, new_col))
        if new_cell.wall:
            return (pos, WALL)
        elif new_cell.goal:
            return ((new_row, new_col), GOAL)
        else:
            return ((new_row, new_col), MOVEMENT)

    def step(self, action: str) -> StepResult:
        if action not in ACTIONS:
            raise ValueError(f"Invalid action: {action}")

        if self.done:
            raise RuntimeError("Cannot step; the goal has already been reached.")

        self.current_step += 1
        new_cell, outcome = self.next_cell(action)

        current_reward = self.reward_config[outcome]
        self.total_reward += current_reward
        self.agent_pos = new_cell

        return StepResult(
            new_state=self.agent_pos,
            reward=current_reward,
            done=self.done,
        )


if __name__ == "__main__":
    import random
    import time

    env = GridWorldEnv()
    max_steps = 100

    state = env.reset()
    for step in range(max_steps):
        console.clear()
        env.render()
        time.sleep(0.5)  # slower for readability

        if env.done:
            break

        action = random.choice(list(ACTIONS.keys()))
        state, reward, done = env.step(action)

    render_heatmap(
        visit_counts=env.visit_counts,
        rows=env.rows,
        cols=env.cols,
    )
