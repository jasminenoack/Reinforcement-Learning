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

from dataclasses import dataclass
from rich.console import Console
from rich.text import Text


console = Console()


@dataclass
class ActionConfig:
    row_movement: int
    col_movement: int


ACTIONS = {
    "up": ActionConfig(-1, 0),
    "down": ActionConfig(1, 0),
    "left": ActionConfig(0, -1),
    "right": ActionConfig(0, 1),
}


@dataclass
class RewardConfiguration:
    step_penalty: float = -0.01
    off_board_penalty: float = -0.1
    goal_reward: float = 1


@dataclass
class StepResult:
    new_state: tuple[int, int]
    reward: float
    done: bool

    def __iter__(self):
        yield self.new_state
        yield self.reward
        yield self.done


class GridWorldEnv:
    def __init__(self) -> None:
        self.rows = 5
        self.cols = 5
        self.start = (0, 0)
        self.goal = (4, 4)
        self.reward_config = RewardConfiguration()
        self.max_steps = 100
        self._setup()

    def _setup(self) -> None:
        self.agent_pos = self.start
        self.total_reward = 0
        self.current_step = 0

    @property
    def done(self) -> bool:
        return self.agent_pos == self.goal or self.current_step >= self.max_steps

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

    def step(self, action: str) -> StepResult:
        if action not in ACTIONS:
            raise ValueError(f"Invalid action: {action}")

        if self.done:
            raise RuntimeError("Cannot step; the goal has already been reached.")

        self.current_step += 1

        action_config = ACTIONS[action]
        new_row = self.agent_pos[0] + action_config.row_movement
        new_col = self.agent_pos[1] + action_config.col_movement
        if not (0 <= new_row < self.rows and 0 <= new_col < self.cols):
            reward = self.reward_config.off_board_penalty
        else:
            self.agent_pos = (new_row, new_col)
            reward = self.reward_config.step_penalty

        if self.done:
            reward += self.reward_config.goal_reward

        self.total_reward += reward

        return StepResult(
            new_state=self.agent_pos,
            reward=reward,
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
        time.sleep(1)  # slower for readability

        if env.done:
            break

        action = random.choice(list(ACTIONS.keys()))
        state, reward, done = env.step(action)
