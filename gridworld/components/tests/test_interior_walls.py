import pytest

from gridworld.components.grid_environment import GridWorldEnv
from gridworld.components.maze_builders import Walls
from gridworld.utils import RIGHT, INTERIOR_WALL


class TestNextCellInteriorWall:
    def test_next_cell_hits_wall_between_cells(self) -> None:
        env = GridWorldEnv(rows=1, cols=2)
        env.agent_pos = (0, 0)
        env.get_cell((0, 0)).walls = Walls(right=True)
        env.get_cell((0, 1)).walls = Walls(left=True)

        before_pos = env.agent_pos
        before_reward = env.total_reward
        before_step = env.current_step
        before_visits = env.visit_counts[(0, 0)]

        pos, effect = env.next_cell(RIGHT)

        assert pos == before_pos
        assert effect == INTERIOR_WALL
        assert env.agent_pos == before_pos
        assert env.total_reward == before_reward
        assert env.current_step == before_step
        assert env.visit_counts[(0, 0)] == before_visits
        assert env.visit_counts[(0, 1)] == 0
