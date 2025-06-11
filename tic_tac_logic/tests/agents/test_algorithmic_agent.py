from tic_tac_logic.agents.algorithmic_agent import AlgorithmicAgent
from tic_tac_logic.env.grid import Grid
from tic_tac_logic.sample_grids import get_testing_sample_grids


class TestAlgorithmicAgent:
    def test_solves_first_sample_grid(self) -> None:
        puzzle = get_testing_sample_grids()[0]
        grid = Grid(puzzle.grid)
        agent = AlgorithmicAgent(rows=len(puzzle.grid), columns=len(puzzle.grid[0]))

        while not grid.won()[0]:
            coord, symbol = agent.act(grid.get_observation())
            result = grid.act(coord, symbol)
            agent.learn(result)
            assert not result.loss_reason
        assert grid.grid == puzzle.expected

    def test_solves_all_sample_grids(self) -> None:
        for puzzle in get_testing_sample_grids():
            grid = Grid(puzzle.grid)
            agent = AlgorithmicAgent(rows=len(puzzle.grid), columns=len(puzzle.grid[0]))
            while not grid.won()[0]:
                coord, symbol = agent.act(grid.get_observation())
                result = grid.act(coord, symbol)
                agent.learn(result)
                assert not result.loss_reason
            assert grid.won()[0]
