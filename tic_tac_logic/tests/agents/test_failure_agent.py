from tic_tac_logic.agents.failure_learning_agent import (
    FailureAgent,
    Move,
    FailureClass,
    ValidPlacement,
)
from tic_tac_logic.sample_grids import get_easy_grid
from tic_tac_logic.constants import X, O, E, StepResult
from tic_tac_logic.env.grid import Grid

# [E, E, E, E, E, E],
# [X, X, E, E, O, E],
# [E, E, O, E, E, O],
# [E, X, E, E, E, E],
# [E, E, E, E, O, E],
# [X, E, X, E, E, E],
# [X, E, O, E, E, X],
# [E, E, E, E, E, E],


class TestLearn:
    def test_learns_from_no_valid_placement(self):
        current_grid = [
            [X, X, O, O],
            [O, E, X, X],
            [O, X, X, O],
            [X, O, O, E],
        ]
        grid = Grid(current_grid)
        agent = FailureAgent(len(current_grid), len(current_grid[0]))
        agent.q_table = {
            "failures": {
                Move(
                    type=FailureClass.ROW,
                    applicable_area="O,  , X, X, ",
                    location=1,
                    symbol=X,
                ),
                Move(
                    type=FailureClass.ROW,
                    applicable_area="O,  , X, X, ",
                    location=1,
                    symbol=O,
                ),
            }
        }
        first_result = grid.act((3, 3), X)
        agent.learn(first_result)
        assert not first_result.loss_reason
        assert agent.q_table == {
            "failures": {
                Move(
                    type=FailureClass.ROW,
                    applicable_area="O,  , X, X, ",
                    location=1,
                    symbol=X,
                ),
                Move(
                    type=FailureClass.ROW,
                    applicable_area="O,  , X, X, ",
                    location=1,
                    symbol=O,
                ),
                Move(
                    type=FailureClass.BOARD,
                    applicable_area=(
                        "X, X, O, O,\n" "O,  , X, X,\n" "O, X, X, O,\n" "X, O, O,  ,\n"
                    ),
                    location=(3, 3),
                    symbol=X,
                ),
            },
        }

    def test_handles_baord_failure(self):
        easy_grid = get_easy_grid()
        easy_grid[1] = ["X", "X", " ", " ", "X", " "]
        easy_grid[5] = ["X", "X", " ", " ", " ", " "]
        grid = Grid(easy_grid)
        sample_result = grid.act((5, 4), X)
        assert sample_result == StepResult(
            coordinate=(5, 4),
            score=-10,
            symbol="X",
            loss_reason="2 rows with complete X with X identical",
            pre_step_grid=[
                [" ", " ", " ", " ", " ", " "],
                ["X", "X", " ", " ", "X", " "],
                [" ", " ", "O", " ", " ", "O"],
                [" ", "X", " ", " ", " ", " "],
                [" ", " ", " ", " ", "O", " "],
                ["X", "X", " ", " ", " ", " "],
                ["X", " ", "O", " ", " ", "X"],
                [" ", " ", " ", " ", " ", " "],
            ],
            grid=[
                [" ", " ", " ", " ", " ", " "],
                ["X", "X", " ", " ", "X", " "],
                [" ", " ", "O", " ", " ", "O"],
                [" ", "X", " ", " ", " ", " "],
                [" ", " ", " ", " ", "O", " "],
                ["X", "X", " ", " ", "X", " "],
                ["X", " ", "O", " ", " ", "X"],
                [" ", " ", " ", " ", " ", " "],
            ],
        )
        agent = FailureAgent(len(easy_grid), len(easy_grid[0]))
        agent.learn(sample_result)
        assert agent.q_table == {
            "failures": {
                Move(
                    type=FailureClass.BOARD,
                    applicable_area=(
                        " ,  ,  ,  ,  ,  ,\n"
                        "X, X,  ,  , X,  ,\n"
                        " ,  , O,  ,  , O,\n"
                        " , X,  ,  ,  ,  ,\n"
                        " ,  ,  ,  , O,  ,\n"
                        "X, X,  ,  ,  ,  ,\n"
                        "X,  , O,  ,  , X,\n"
                        " ,  ,  ,  ,  ,  ,\n"
                    ),
                    location=(5, 4),
                    symbol=X,
                ),
            }
        }

    def test_learns_from_broken_column(self):
        easy_grid = get_easy_grid()
        grid = Grid(easy_grid)
        sample_result = grid.act((7, 0), X)
        assert sample_result == StepResult(
            coordinate=(7, 0),
            score=-10,
            symbol="X",
            loss_reason="TOO_MANY_X_TOGETHER_IN_COLUMN",
            pre_step_grid=[
                [" ", " ", " ", " ", " ", " "],
                ["X", "X", " ", " ", "O", " "],
                [" ", " ", "O", " ", " ", "O"],
                [" ", "X", " ", " ", " ", " "],
                [" ", " ", " ", " ", "O", " "],
                ["X", " ", "X", " ", " ", " "],
                ["X", " ", "O", " ", " ", "X"],
                [" ", " ", " ", " ", " ", " "],
            ],
            grid=[
                [" ", " ", " ", " ", " ", " "],
                ["X", "X", " ", " ", "O", " "],
                [" ", " ", "O", " ", " ", "O"],
                [" ", "X", " ", " ", " ", " "],
                [" ", " ", " ", " ", "O", " "],
                ["X", " ", "X", " ", " ", " "],
                ["X", " ", "O", " ", " ", "X"],
                ["X", " ", " ", " ", " ", " "],
            ],
        )
        agent = FailureAgent(len(easy_grid), len(easy_grid[0]))

        agent.learn(sample_result)
        assert agent.q_table == {
            "failures": {
                Move(
                    type=FailureClass.COLUMN,
                    applicable_area=" , X,  ,  ,  , X, X,  , ",
                    location=7,
                    symbol=X,
                )
            }
        }

    def test_learns_from_too_many_x_in_row(self):
        easy_grid = get_easy_grid()
        grid = Grid(easy_grid)

        sample_result = grid.act((1, 2), X)
        assert sample_result == StepResult(
            coordinate=(1, 2),
            score=-10,
            symbol="X",
            loss_reason="TOO_MANY_X_TOGETHER_IN_ROW",
            pre_step_grid=[
                [" ", " ", " ", " ", " ", " "],
                ["X", "X", " ", " ", "O", " "],
                [" ", " ", "O", " ", " ", "O"],
                [" ", "X", " ", " ", " ", " "],
                [" ", " ", " ", " ", "O", " "],
                ["X", " ", "X", " ", " ", " "],
                ["X", " ", "O", " ", " ", "X"],
                [" ", " ", " ", " ", " ", " "],
            ],
            grid=[
                [" ", " ", " ", " ", " ", " "],
                ["X", "X", "X", " ", "O", " "],
                [" ", " ", "O", " ", " ", "O"],
                [" ", "X", " ", " ", " ", " "],
                [" ", " ", " ", " ", "O", " "],
                ["X", " ", "X", " ", " ", " "],
                ["X", " ", "O", " ", " ", "X"],
                [" ", " ", " ", " ", " ", " "],
            ],
        )
        agent = FailureAgent(len(easy_grid), len(easy_grid[0]))

        agent.learn(sample_result)
        assert agent.q_table == {
            "failures": {
                Move(
                    type=FailureClass.ROW,
                    applicable_area="X, X,  ,  , O,  , ",
                    location=2,
                    symbol=X,
                )
            }
        }


class TestGetValidPlacements:
    def test_all_empty_spaces_have_two_valid_placements_to_start(self):
        easy_grid = get_easy_grid()
        grid = Grid(easy_grid)
        agent = FailureAgent(len(easy_grid), len(easy_grid[0]))
        valid_placements = agent.get_valid_placements(grid.grid)
        assert len(valid_placements) == 72

    def test_removes_moves_that_have_an_active_failure(self):
        easy_grid = get_easy_grid()
        grid = Grid(easy_grid)
        result = grid.act((1, 2), X)
        agent = FailureAgent(len(easy_grid), len(easy_grid[0]))
        agent.learn(result)
        grid.reset()
        valid_placements = agent.get_valid_placements(grid.grid)
        assert len(valid_placements) == 71
        assert ValidPlacement(coordinate=(1, 2), symbol=X) not in valid_placements

    def test_removes_identical_failure_in_another_row(self):
        easy_grid = get_easy_grid()
        easy_grid[3] = ["X", "X", " ", " ", "O", " "]
        grid = Grid(easy_grid)

        result = grid.act((1, 2), X)
        agent = FailureAgent(len(easy_grid), len(easy_grid[0]))
        agent.learn(result)
        grid.reset()
        valid_placements = agent.get_valid_placements(grid.grid)
        assert len(valid_placements) == 66
        assert ValidPlacement(coordinate=(1, 2), symbol=X) not in valid_placements
        assert ValidPlacement(coordinate=(3, 2), symbol=X) not in valid_placements

    def test_works_with_O(self):
        easy_grid = get_easy_grid()
        grid = Grid(easy_grid)
        agent = FailureAgent(len(easy_grid), len(easy_grid[0]))
        result = grid.act((1, 2), O)
        before_valid_placements = agent.get_valid_placements(grid.grid)
        assert len(before_valid_placements) == 70
        agent.learn(result)
        result = grid.act((1, 3), O)
        agent.learn(result)
        grid.reset()
        assert agent.q_table == {
            "failures": {
                Move(
                    type=FailureClass.ROW,
                    applicable_area="X, X, O,  , O,  , ",
                    location=3,
                    symbol=O,
                ),
            }
        }
        result = grid.act((1, 2), O)
        valid_placements = agent.get_valid_placements(grid.grid)
        assert len(valid_placements) == 69
        assert ValidPlacement(coordinate=(1, 3), symbol=O) not in valid_placements

    def test_removes_if_broken_in_column(self):
        easy_grid = get_easy_grid()
        grid = Grid(easy_grid)
        agent = FailureAgent(len(easy_grid), len(easy_grid[0]))
        assert len(agent.get_valid_placements(grid.grid)) == 72
        sample_result = grid.act((7, 0), X)
        agent.learn(sample_result)
        assert len(agent.get_valid_placements(grid.grid)) == 70
        grid.reset()
        assert len(agent.get_valid_placements(grid.grid)) == 71

    def test_removes_baord_failure(self):
        easy_grid = get_easy_grid()
        easy_grid[1] = ["X", "X", " ", " ", "X", " "]
        easy_grid[5] = ["X", "X", " ", " ", " ", " "]
        grid = Grid(easy_grid)
        agent = FailureAgent(len(easy_grid), len(easy_grid[0]))
        assert len(agent.get_valid_placements(grid.grid)) == 72
        sample_result = grid.act((5, 4), X)
        agent.learn(sample_result)
        assert len(agent.get_valid_placements(grid.grid)) == 70
        grid.reset()
        assert len(agent.get_valid_placements(grid.grid)) == 71
