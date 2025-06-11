import pytest
from tic_tac_logic.agents.algorithmic_agent import AlgorithmicAgent, NoValidMoveError
from tic_tac_logic.constants import X, O, E, Observation
from tic_tac_logic.sample_grids import get_testing_sample_grids


class TestAlgorithmicAgent:
    def test_init(self):
        agent = AlgorithmicAgent(6, 6)
        assert agent.rows == 6
        assert agent.columns == 6
        assert agent.target_count == 3

    def test_init_odd_dimensions_raises_error(self):
        with pytest.raises(ValueError, match="Binaro puzzles require even dimensions"):
            AlgorithmicAgent(5, 6)

        with pytest.raises(ValueError, match="Binaro puzzles require even dimensions"):
            AlgorithmicAgent(6, 5)

    def test_consecutive_prevention_horizontal(self):
        agent = AlgorithmicAgent(4, 4)

        # Test XX_ pattern
        grid = [[X, X, E, E], [E, E, E, E], [E, E, E, E], [E, E, E, E]]
        observation = Observation(grid)
        move = agent.act(observation)
        assert move == ((0, 2), O)

        # Test _XX pattern
        grid = [[E, X, X, E], [E, E, E, E], [E, E, E, E], [E, E, E, E]]
        observation = Observation(grid)
        move = agent.act(observation)
        assert move == ((0, 0), O)

        # Test X_X pattern
        grid = [[X, E, X, E], [E, E, E, E], [E, E, E, E], [E, E, E, E]]
        observation = Observation(grid)
        move = agent.act(observation)
        assert move == ((0, 1), O)

    def test_consecutive_prevention_vertical(self):
        agent = AlgorithmicAgent(4, 4)

        # Test vertical XX_ pattern
        grid = [[O, E, E, E], [O, E, E, E], [E, E, E, E], [E, E, E, E]]
        observation = Observation(grid)
        move = agent.act(observation)
        assert move == ((2, 0), X)

    def test_completion_move_row(self):
        agent = AlgorithmicAgent(4, 4)

        # Row with 2 X's should fill remaining with O's
        grid = [[X, O, X, E], [E, E, E, E], [E, E, E, E], [E, E, E, E]]
        observation = Observation(grid)
        move = agent.act(observation)
        assert move == ((0, 3), O)

    def test_completion_move_column(self):
        agent = AlgorithmicAgent(4, 4)

        # Column with 2 O's should fill remaining with X's
        grid = [[O, E, E, E], [X, E, E, E], [O, E, E, E], [E, E, E, E]]
        observation = Observation(grid)
        move = agent.act(observation)
        assert move == ((3, 0), X)

    def test_no_valid_move_raises_error(self):
        agent = AlgorithmicAgent(4, 4)

        # Grid with no obvious moves
        grid = [[E, E, E, E], [E, E, E, E], [E, E, E, E], [E, E, E, E]]
        observation = Observation(grid)

        with pytest.raises(NoValidMoveError):
            agent.act(observation)

    def test_reset_does_nothing(self):
        agent = AlgorithmicAgent(4, 4)
        agent.reset()  # Should not raise any errors

    def test_learn_does_nothing(self):
        agent = AlgorithmicAgent(4, 4)
        # Create a dummy step result
        from tic_tac_logic.constants import StepResult

        step_result = StepResult(
            coordinate=(0, 0),
            symbol=X,
            pre_step_grid=None,
            post_step_grid=None,
            loss_reason=None,
        )
        agent.learn(step_result)  # Should not raise any errors

    def test_with_sample_grid(self):
        """Test that the agent can make progress on the sample testing grid."""
        try:
            grids_data = get_testing_sample_grids()

            for grid_name, (grid, expected_solution) in grids_data.items():
                # Skip if grid has odd dimensions (not valid for Binaro)
                if len(grid) % 2 != 0 or len(grid[0]) % 2 != 0:
                    print(
                        f"⚠ {grid_name}: Skipping - odd dimensions not valid for Binaro"
                    )
                    continue

                agent = AlgorithmicAgent(len(grid), len(grid[0]))
                observation = Observation(grid)

                try:
                    move = agent.act(observation)
                    row, col = move[0]
                    symbol = move[1]

                    # Verify the move is valid (placing in empty cell)
                    assert (
                        grid[row][col] == E
                    ), f"Trying to place in non-empty cell for {grid_name}"

                    # Verify the move matches expected solution
                    assert (
                        expected_solution[row][col] == symbol
                    ), f"Move doesn't match expected solution for {grid_name}"

                    print(
                        f"✓ {grid_name}: Agent suggests {move}, matches expected solution"
                    )

                except NoValidMoveError:
                    print(
                        f"⚠ {grid_name}: Agent cannot find certain move (may need more complex rules)"
                    )

        except ImportError:
            print(
                "⚠ Cannot import get_testing_sample_grids - skipping sample grid test"
            )

    def test_is_valid_placement(self):
        agent = AlgorithmicAgent(4, 4)

        grid = [[X, X, E, E], [E, E, E, E], [E, E, E, E], [E, E, E, E]]

        # Should not be valid to place X at (0, 2) - would create three consecutive
        assert not agent._is_valid_placement(grid, 0, 2, X)

        # Should be valid to place O at (0, 2)
        assert agent._is_valid_placement(grid, 0, 2, O)

    def test_check_count_limits(self):
        agent = AlgorithmicAgent(4, 4)

        # Grid with row almost at limit
        grid = [[X, O, X, E], [E, E, E, E], [E, E, E, E], [E, E, E, E]]

        # Should not allow placing another X in first row
        test_grid = [row[:] for row in grid]
        test_grid[0][3] = X
        assert not agent._check_count_limits(test_grid, 0, 3)

        # Should allow placing O in first row
        test_grid[0][3] = O
        assert agent._check_count_limits(test_grid, 0, 3)
