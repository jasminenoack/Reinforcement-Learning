import pytest
from tic_tac_logic.agents.algorithmic_agent import AlgorithmicAgent
from tic_tac_logic.env.grid import Grid
from tic_tac_logic.sample_grids import get_one_off_grid, get_testing_sample_grids
from tic_tac_logic.constants import X


class TestAlgorithmicAgent:
    def test_act_returns_optimal_move(self):
        grid_data = get_one_off_grid()
        grid = Grid(grid_data)
        agent = AlgorithmicAgent(len(grid_data), len(grid_data[0]))
        action = agent.act(grid.get_observation())
        assert action == ((1, 1), X)

    def test_solves_first_test_puzzle(self):
        puzzle = get_testing_sample_grids()[0]
        grid = Grid(puzzle.grid)
        agent = AlgorithmicAgent(len(puzzle.grid), len(puzzle.grid[0]))
        steps = 0
        while not grid.won()[0]:
            action = agent.act(grid.get_observation())
            grid.act(*action)
            steps += 1
            assert steps <= len(puzzle.grid) * len(puzzle.grid[0])
        assert grid.grid == puzzle.expected
