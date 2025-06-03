from tic_tac_logic.agents.failure_learning_agent import (
    FailureAgent,
    Move,
    FailureClass,
    ValidPlacement,
)
from tic_tac_logic.sample_grids import get_easy_grid
from tic_tac_logic.constants import X, O, StepResult
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
        agent = FailureAgent(easy_grid)

        agent.learn(sample_result)
        assert agent.q_table == {
            "failures": {
                Move(
                    type=FailureClass.ROW,
                    applicable_area="X, X,  ,  , O,  ,",
                    location=2,
                    symbol=X,
                )
            }
        }


class TestGetValidPlacements:
    def test_all_empty_spaces_have_two_valid_placements_to_start(self):
        easy_grid = get_easy_grid()
        grid = Grid(easy_grid)
        agent = FailureAgent(easy_grid)
        valid_placements = agent.get_valid_placements(grid.grid)
        assert len(valid_placements) == 72

    def test_removes_moves_that_have_an_active_failure(self):
        easy_grid = get_easy_grid()
        grid = Grid(easy_grid)
        result = grid.act((1, 2), X)
        agent = FailureAgent(easy_grid)
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
        agent = FailureAgent(easy_grid)
        agent.learn(result)
        grid.reset()
        valid_placements = agent.get_valid_placements(grid.grid)
        assert len(valid_placements) == 66
        assert ValidPlacement(coordinate=(1, 2), symbol=X) not in valid_placements
        assert ValidPlacement(coordinate=(3, 2), symbol=X) not in valid_placements

    def test_works_with_O(self):
        easy_grid = get_easy_grid()
        grid = Grid(easy_grid)
        agent = FailureAgent(easy_grid)
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
                    applicable_area="X, X, O,  , O,  ,",
                    location=3,
                    symbol=O,
                ),
            }
        }
        result = grid.act((1, 2), O)
        valid_placements = agent.get_valid_placements(grid.grid)
        assert len(valid_placements) == 69
        print(before_valid_placements - valid_placements)
        assert ValidPlacement(coordinate=(1, 3), symbol=O) not in valid_placements


# class TestAct:
#     def test_will_not_generate_an_action_that_is_already_assumed_to_fail(self):
#         easy_grid = get_easy_grid()
#         grid = Grid(easy_grid)

#         agent = FailureAgent(easy_grid)
#         agent.q_table = {
#             "failures": {
#                 Move(
#                     type=FailureClass.ROW,
#                     applicable_area="X, X, X,  , O,  ,",
#                     coordinate=(1, 2),
#                     symbol=X,
#                 )
#             }
#         }
#         action = agent.act(grid.grid, X)
#         assert action == (0, 0)
