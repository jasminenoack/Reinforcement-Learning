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
from rich.console import Console
from rich.text import Text

from gridworld.components.maze_builders import Entry, Walls
from gridworld.utils import (
    DOWN,
    GOAL,
    INTERIOR_WALL,
    LEFT,
    MOVEMENT,
    OFF_BOARD,
    REVERSED_ACTIONS,
    RIGHT,
    UP,
    OBSTACLE,
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
    OBSTACLE: -10,
    GOAL: 100,
    MOVEMENT: -1,
    INTERIOR_WALL: -10,
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
    _obstacle: bool = False
    visited: int = 0
    walls: Walls | None = None

    @property
    def obstacle(self) -> bool:
        return self._obstacle

    @obstacle.setter
    def obstacle(self, value: bool) -> None:
        if self.agent:
            raise ValueError("Cannot set obstacle on a cell with an agent.")
        if self.goal:
            raise ValueError("Cannot set obstacle on a cell with a goal.")
        self._obstacle = value

    def has_door(self, direction: str) -> bool:
        if self.walls is None:
            return True
        if direction == UP:
            return not self.walls.up
        elif direction == DOWN:
            return not self.walls.down
        elif direction == LEFT:
            return not self.walls.left
        elif direction == RIGHT:
            return not self.walls.right
        else:
            raise ValueError(f"Invalid direction: {direction}")


class GridWorldEnv:
    def __init__(
        self,
        *,
        max_steps: int = 100,
        rows: int = 5,
        cols: int = 5,
        grid: list[list[Entry]] = None,
    ) -> None:
        found_start = None
        found_goal = None
        if grid:
            self.rows = len(grid)
            self.cols = len(grid[0])

            for i, row in enumerate(grid):
                for j, entry in enumerate(row):
                    if entry.start:
                        if found_start is not None:
                            raise ValueError("Multiple start positions found.")
                        found_start = (i, j)
                    if entry.goal:
                        if found_goal is not None:
                            raise ValueError("Multiple goal positions found.")
                        found_goal = (i, j)

        else:
            self.rows = rows
            self.cols = cols

        self.reward_config = STEP_RESULT_REWARD
        self.max_steps = max_steps

        if found_start is None:
            self.start = (0, 0)
        else:
            self.start = found_start
        if found_goal is None:
            self.goal = (self.rows - 1, self.cols - 1)
        else:
            self.goal = found_goal
        self._config_grid = grid
        self._setup()

    def _setup(self) -> None:
        self.total_reward = 0
        self.current_step = 0
        if self._config_grid:
            self.grid = []
            for row in self._config_grid:
                row_cells = []
                for entry in row:
                    cell = Cell(
                        agent=entry.start,
                        goal=entry.goal,
                        _obstacle=entry.obstacle,
                        walls=entry.walls,
                    )
                    row_cells.append(cell)
                self.grid.append(row_cells)
        else:
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

        current_cell = self.get_cell(pos)
        if not current_cell.has_door(action):
            return (pos, INTERIOR_WALL)

        new_row = pos[0] + action_config.row_movement
        new_col = pos[1] + action_config.col_movement
        if not (0 <= new_row < self.rows and 0 <= new_col < self.cols):
            return (pos, OFF_BOARD)

        new_cell = self.get_cell((new_row, new_col))
        revered_action = REVERSED_ACTIONS[action]
        if not new_cell.has_door(revered_action):
            return (pos, INTERIOR_WALL)

        if new_cell.obstacle:
            return (pos, OBSTACLE)
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
