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

Future extensions may include visualization tools (e.g., heatmaps, path tracking),
Q-table inspection, and dynamic difficulty.

Example usage:
    env = GridworldEnv(rows=5, cols=5)
    env.reset()
    env.render()
"""


class GridWorldEnv:
    def __init__(self, rows=5, cols=5):
        self.rows = rows
        self.cols = cols
