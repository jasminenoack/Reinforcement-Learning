import pytest

from gridworld.components.maze_builders import Entry, Walls
from gridworld.utils import DOWN, GOAL, LEFT, MOVEMENT, OFF_BOARD, RIGHT, UP, OBSTACLE
from ..grid_environment import Cell, GridWorldEnv


class TestGridWorldEnvInit:
    def test_can_create_env(self):
        env = GridWorldEnv()
        assert env.rows == 5
        assert env.cols == 5
        assert env.start == (0, 0)
        assert env.goal == (4, 4)
        assert env.total_reward == 0
        assert env.agent_pos == (0, 0)
        assert env.done is False
        assert env.max_steps == 100
        assert env.current_step == 0
        assert len(env.grid) == 5
        assert len(env.grid[0]) == 5
        assert env.grid[0][0] == Cell(
            agent=True, goal=False, _obstacle=False, visited=1
        )
        assert env.grid[4][4] == Cell(
            agent=False, goal=True, _obstacle=False, visited=0
        )
        assert env.grid[0][1] == Cell(
            agent=False, goal=False, _obstacle=False, visited=0
        )
        assert env.visit_counts[(0, 0)] == 1
        assert sum(env.visit_counts.values()) == 1

    def test_can_set_max_steps(self):
        env = GridWorldEnv(max_steps=37)
        assert env.max_steps == 37

    def test_can_set_rows_and_cols(self):
        env = GridWorldEnv(rows=20, cols=20)
        assert env.start == (0, 0)
        assert env.goal == (19, 19)

    def test_can_be_not_square(self):
        env = GridWorldEnv(rows=10, cols=5)
        assert env.start == (0, 0)
        assert env.goal == (9, 4)

    def test_no_obstacless_by_default(self):
        env = GridWorldEnv()
        for row in env.grid:
            for cell in row:
                assert cell.obstacle is False

    def test_pulls_start_and_goal_from_grid(self):
        grid = [
            [
                Entry(
                    start=False,
                    goal=False,
                    obstacle=False,
                ),
                Entry(
                    start=False,
                    goal=False,
                    obstacle=False,
                ),
                Entry(
                    start=False,
                    goal=True,
                    obstacle=False,
                ),
            ],
            [
                Entry(
                    start=True,
                    goal=False,
                    obstacle=False,
                ),
                Entry(
                    start=False,
                    goal=False,
                    obstacle=False,
                ),
                Entry(
                    start=False,
                    goal=False,
                    obstacle=False,
                ),
            ],
        ]
        env = GridWorldEnv(grid=grid)
        assert env.start == (1, 0)
        assert env.goal == (0, 2)
        assert env.rows == 2
        assert env.cols == 3

    def test_handles_no_start_finish_in_grid(self):
        grid = [
            [
                Entry(
                    start=False,
                    goal=False,
                    obstacle=False,
                ),
                Entry(
                    start=False,
                    goal=False,
                    obstacle=False,
                ),
                Entry(
                    start=False,
                    goal=False,
                    obstacle=False,
                ),
            ],
            [
                Entry(
                    start=False,
                    goal=False,
                    obstacle=False,
                ),
                Entry(
                    start=False,
                    goal=False,
                    obstacle=False,
                ),
                Entry(
                    start=False,
                    goal=False,
                    obstacle=False,
                ),
            ],
        ]
        env = GridWorldEnv(grid=grid)
        assert env.start == (0, 0)
        assert env.goal == (1, 2)
        assert env.rows == 2
        assert env.cols == 3

    def test_can_pass_a_grid_with_obstacles(self):
        grid = [
            [
                Entry(
                    start=True,
                    obstacle=False,
                ),
                Entry(
                    obstacle=False,
                ),
                Entry(
                    obstacle=True,
                ),
            ],
            [
                Entry(
                    obstacle=False,
                ),
                Entry(
                    obstacle=False,
                ),
                Entry(
                    obstacle=False,
                ),
            ],
            [
                Entry(
                    obstacle=True,
                ),
                Entry(
                    obstacle=False,
                ),
                Entry(
                    obstacle=False,
                ),
            ],
        ]
        env = GridWorldEnv(grid=grid)
        assert env.get_cell((0, 0)).obstacle is False
        assert env.get_cell((0, 1)).obstacle is False
        assert env.get_cell((0, 2)).obstacle is True
        assert env.get_cell((1, 0)).obstacle is False
        assert env.get_cell((1, 1)).obstacle is False
        assert env.get_cell((1, 2)).obstacle is False
        assert env.get_cell((2, 0)).obstacle is True
        assert env.get_cell((2, 1)).obstacle is False
        assert env.get_cell((2, 2)).obstacle is False

    def test_can_pass_a_grid_with_walls(self):
        grid = [
            [
                Entry(
                    walls=Walls(up=True, down=False, left=False, right=False),
                ),
                Entry(
                    walls=Walls(up=False, down=False, left=False, right=False),
                ),
                Entry(
                    walls=Walls(up=False, down=False, left=False, right=False),
                ),
            ],
            [
                Entry(
                    walls=Walls(up=False, down=False, left=False, right=False),
                ),
                Entry(
                    walls=Walls(up=False, down=False, left=False, right=False),
                ),
                Entry(
                    walls=Walls(up=False, down=False, left=False, right=False),
                ),
            ],
            [
                Entry(
                    walls=Walls(up=False, down=False, left=False, right=False),
                ),
                Entry(
                    walls=Walls(up=False, down=False, left=False, right=False),
                ),
                Entry(
                    walls=Walls(up=False, down=False, left=False, right=False),
                ),
            ],
        ]
        env = GridWorldEnv(grid=grid)
        assert env.get_cell((0, 0)).walls == Walls(
            up=True,
            down=False,
            left=False,
            right=False,
        )
        assert env.get_cell((0, 1)).walls == Walls(
            up=False,
            down=False,
            left=False,
            right=False,
        )


class TestReset:
    def test_moves_robot_back_to_start(self):
        env = GridWorldEnv()
        env.agent_pos = (4, 4)
        env.total_reward = 1
        env.current_step = 10
        pos = env.reset()
        assert env.agent_pos == (0, 0)
        assert env.done is False
        assert env.total_reward == 0
        assert pos == (0, 0)
        assert env.current_step == 0
        assert env.visit_counts[(0, 0)] == 1
        assert sum(env.visit_counts.values()) == 1


class TestRender:
    def test_renders_grid_with_agent_and_goal(self, capsys):
        env = GridWorldEnv()
        env.agent_pos = (2, 2)
        env.render()
        captured = capsys.readouterr()
        lines = captured.out.splitlines()
        assert lines[-6] == ""
        assert lines[-5] == ". . . . . "
        assert lines[-4] == ". . . . . "
        assert lines[-3] == ". . A . . "
        assert lines[-2] == ". . . . . "
        assert lines[-1] == ". . . . G "


class TestDone:
    def test_done_is_false_when_not_at_goal(self):
        env = GridWorldEnv()
        env.agent_pos = (2, 2)
        assert env.done is False

    def test_done_is_true_when_at_goal(self):
        env = GridWorldEnv()
        env.agent_pos = (4, 4)
        assert env.done is True

    def test_not_done_at_start(self):
        env = GridWorldEnv()
        env.agent_pos = (0, 0)
        assert env.done is False

    def test_done_at_max_steps(self):
        env = GridWorldEnv()
        env.agent_pos = (0, 0)
        env.current_step = 100
        assert env.done is True


class TestReachedGoal:
    def test_reached_goal_is_false_when_not_at_goal(self):
        env = GridWorldEnv()
        env.agent_pos = (2, 2)
        assert env.reached_goal is False

    def test_reached_goal_is_true_when_at_goal(self):
        env = GridWorldEnv()
        env.agent_pos = (4, 4)
        assert env.reached_goal is True

    def test_reached_goal_at_start(self):
        env = GridWorldEnv()
        env.agent_pos = (0, 0)
        assert env.reached_goal is False


class TestGetState:
    def test_get_state(self):
        env = GridWorldEnv()
        env.agent_pos = (2, 2)
        state = env.get_state()
        assert state == (2, 2)


class TestStep:
    def test_can_move_agent_up(self):
        env = GridWorldEnv()
        env.agent_pos = (2, 2)
        action = "up"
        new_state, reward, done = env.step(action)
        assert new_state == (1, 2)
        assert reward == -1
        assert done is False

    def test_can_move_agent_down(self):
        env = GridWorldEnv()
        env.agent_pos = (2, 2)
        action = "down"
        new_state, reward, done = env.step(action)
        assert new_state == (3, 2)
        assert reward == -1
        assert done is False

    def test_can_move_agent_left(self):
        env = GridWorldEnv()
        env.agent_pos = (2, 2)
        action = "left"
        new_state, reward, done = env.step(action)
        assert new_state == (2, 1)
        assert reward == -1
        assert done is False

    def test_can_move_agent_right(self):
        env = GridWorldEnv()
        env.agent_pos = (2, 2)
        action = "right"
        new_state, reward, done = env.step(action)
        assert new_state == (2, 3)
        assert reward == -1
        assert done is False

    def test_agent_stays_if_off_board_up(self):
        env = GridWorldEnv()
        env.agent_pos = (0, 2)
        action = "up"
        new_state, reward, done = env.step(action)
        assert new_state == (0, 2)
        assert reward == -10
        assert done is False

    def test_agent_stays_if_off_board_down(self):
        env = GridWorldEnv()
        env.agent_pos = (4, 2)
        action = "down"
        new_state, reward, done = env.step(action)
        assert new_state == (4, 2)
        assert reward == -10
        assert done is False

    def test_agent_stays_if_off_board_left(self):
        env = GridWorldEnv()
        env.agent_pos = (2, 0)
        action = "left"
        new_state, reward, done = env.step(action)
        assert new_state == (2, 0)
        assert reward == -10
        assert done is False

    def test_agent_stays_if_off_board_right(self):
        env = GridWorldEnv()
        env.agent_pos = (2, 4)
        action = "right"
        new_state, reward, done = env.step(action)
        assert new_state == (2, 4)
        assert reward == -10
        assert done is False

    def test_agent_is_happy_when_reaching_goal(self):
        env = GridWorldEnv()
        env.agent_pos = (3, 4)
        action = "down"
        new_state, reward, done = env.step(action)
        assert new_state == (4, 4)
        assert reward == 100
        assert done is True

    def test_errors_if_done(self):
        env = GridWorldEnv()
        env.agent_pos = (4, 4)
        action = "up"
        with pytest.raises(Exception):
            env.step(action)

    def test_errors_if_invalid_action(self):
        env = GridWorldEnv()
        env.agent_pos = (2, 2)
        action = "invalid_action"
        with pytest.raises(Exception):
            env.step(action)

    def test_sums_rewards(self):
        env = GridWorldEnv()
        env.agent_pos = (2, 2)
        action = "up"
        new_state, reward, done = env.step(action)
        assert new_state == (1, 2)
        assert reward == -1
        assert done is False
        assert env.total_reward == -1
        assert env.current_step == 1
        action = "up"
        new_state, reward, done = env.step(action)
        assert new_state == (0, 2)
        assert reward == -1
        assert done is False
        assert env.total_reward == -2
        assert env.current_step == 2
        action = "up"
        new_state, reward, done = env.step(action)
        assert new_state == (0, 2)
        assert reward == -10
        assert done is False
        assert env.total_reward == -12
        assert env.current_step == 3

    def test_increments_step_count(self):
        env = GridWorldEnv()
        env.agent_pos = (2, 2)
        action = "up"
        new_state, reward, done = env.step(action)
        assert new_state == (1, 2)
        assert reward == -1
        assert done is False
        assert env.current_step == 1

    def test_increments_visit_count(self):
        env = GridWorldEnv()
        env.step(DOWN)
        env.step(UP)

        assert env.visit_counts == {(1, 0): 1, (0, 0): 2}

    def adds_visit_count_when_walking_into_obstacle(self):
        env = GridWorldEnv()
        env.agent_pos = (0, 0)
        action = "up"
        new_state, reward, done = env.step(action)
        assert new_state == (0, 0)
        assert reward == -10
        assert done is False
        assert env.visit_counts == {(0, 0): 2}

    def test_does_not_get_goal_reward_if_max_steps_not_at_goal(self):
        env = GridWorldEnv()
        env.agent_pos = (3, 4)
        env.current_step = 99
        action = "left"
        new_state, reward, done = env.step(action)
        assert new_state == (3, 3)
        assert reward == -1
        assert done is True
        assert env.total_reward == -1

    def test_handles_walking_into_wall_up(self):
        env = GridWorldEnv()
        env.agent_pos = (1, 1)
        env.get_cell((1, 1)).walls = Walls(
            up=True,
        )
        action = "up"
        new_state, reward, done = env.step(action)
        assert new_state == (1, 1)
        assert reward == -10
        assert done is False
        assert env.visit_counts[(1, 1)] == 2
        assert env.visit_counts[(0, 1)] == 0

    def test_handles_walking_into_wall_down(self):
        env = GridWorldEnv()
        env.agent_pos = (1, 1)
        env.get_cell((1, 1)).walls = Walls(
            down=True,
        )
        action = "down"
        new_state, reward, done = env.step(action)
        assert new_state == (1, 1)
        assert reward == -10
        assert done is False
        assert env.visit_counts[(1, 1)] == 2
        assert env.visit_counts[(2, 1)] == 0

    def test_handles_walking_into_wall_left(self):
        env = GridWorldEnv()
        env.agent_pos = (1, 1)
        env.get_cell((1, 1)).walls = Walls(
            left=True,
        )
        action = "left"
        new_state, reward, done = env.step(action)
        assert new_state == (1, 1)
        assert reward == -10
        assert done is False
        assert env.visit_counts[(1, 1)] == 2
        assert env.visit_counts[(1, 0)] == 0

    def test_handles_walking_into_wall_right(self):
        env = GridWorldEnv()
        env.agent_pos = (1, 2)
        env.get_cell((1, 2)).walls = Walls(
            right=True,
        )
        action = "right"
        new_state, reward, done = env.step(action)
        assert new_state == (1, 2)
        assert reward == -10
        assert done is False
        assert env.visit_counts[(1, 2)] == 2
        assert env.visit_counts[(1, 3)] == 0

    def test_other_cell_has_wall_up(self):
        env = GridWorldEnv()
        env.agent_pos = (1, 1)
        env.get_cell((0, 1)).walls = Walls(
            down=True,
        )
        action = "up"
        new_state, reward, done = env.step(action)
        assert new_state == (1, 1)
        assert reward == -10
        assert done is False
        assert env.visit_counts[(1, 1)] == 2
        assert env.visit_counts[(0, 1)] == 0

    def test_other_cell_has_wall_down(self):
        env = GridWorldEnv()
        env.agent_pos = (1, 1)
        env.get_cell((2, 1)).walls = Walls(
            up=True,
        )
        action = "down"
        new_state, reward, done = env.step(action)
        assert new_state == (1, 1)
        assert reward == -10
        assert done is False
        assert env.visit_counts[(1, 1)] == 2
        assert env.visit_counts[(2, 1)] == 0

    def test_other_cell_has_wall_left(self):
        env = GridWorldEnv()
        env.agent_pos = (1, 1)
        env.get_cell((1, 0)).walls = Walls(
            right=True,
        )
        action = "left"
        new_state, reward, done = env.step(action)
        assert new_state == (1, 1)
        assert reward == -10
        assert done is False
        assert env.visit_counts[(1, 1)] == 2
        assert env.visit_counts[(1, 0)] == 0

    def test_other_cell_has_wall_right(self):
        env = GridWorldEnv()
        env.agent_pos = (1, 1)
        env.get_cell((1, 2)).walls = Walls(
            left=True,
        )
        action = "right"
        new_state, reward, done = env.step(action)
        assert new_state == (1, 1)
        assert reward == -10
        assert done is False
        assert env.visit_counts[(1, 1)] == 2
        assert env.visit_counts[(1, 2)] == 0


class TestVisitCounts:
    def test_visit_counts(self):
        env = GridWorldEnv()
        env.agent_pos = (2, 2)
        action = "up"
        new_state, reward, done = env.step(action)
        assert new_state == (1, 2)
        assert reward == -1
        assert done is False
        assert env.visit_counts[(0, 0)] == 1
        assert env.visit_counts[(1, 2)] == 1


class TestFindAgentPosition:
    def test_find_agent_position(self):
        env = GridWorldEnv()
        env.agent_pos = (2, 2)
        pos = env.find_agent_position()
        assert pos == (2, 2)

    def test_find_agent_position_not_found(self):
        env = GridWorldEnv()
        env.agent_pos = (0, 0)
        pos = env.find_agent_position()
        assert pos == (0, 0)


class TestAgentPos:
    def test_agent_pos(self):
        env = GridWorldEnv()
        env.agent_pos = (2, 2)
        assert env.agent_pos == (2, 2)

    def test_agent_pos_setter(self):
        env = GridWorldEnv()
        env.agent_pos = (2, 2)
        env.agent_pos = (3, 3)
        assert env.agent_pos == (3, 3)


class TestSetAgentPos:
    def test_set_agent_pos(self):
        env = GridWorldEnv()
        env.agent_pos = (2, 2)
        assert env.agent_pos == (2, 2)
        assert env.visit_counts[(2, 2)] == 1


class TestGetCell:
    def test_get_cell(self):
        env = GridWorldEnv()
        cell = env.get_cell((2, 2))
        assert cell.agent is False
        assert cell.goal is False
        assert cell.obstacle is False
        assert cell.visited == 0

    def test_get_goal_cell(self):
        env = GridWorldEnv()
        cell = env.get_cell((4, 4))
        assert cell.agent is False
        assert cell.goal is True
        assert cell.obstacle is False
        assert cell.visited == 0

    def test_get_agent_cell(self):
        env = GridWorldEnv()
        cell = env.get_cell((0, 0))
        assert cell.agent is True
        assert cell.goal is False
        assert cell.obstacle is False
        assert cell.visited == 1


class TestFindGoalPosition:
    def test_find_goal_position(self):
        env = GridWorldEnv()
        pos = env.find_goal_position()
        assert pos == (4, 4)


class TestVisit:
    def test_visit(self):
        env = GridWorldEnv()
        env.agent_pos = (2, 2)
        env.visit(new_pos=(1, 2))
        assert env.grid[1][2].visited == 1
        assert env.grid[0][0].visited == 1
        assert env.visit_counts[(1, 2)] == 1
        assert env.visit_counts[(0, 0)] == 1


class TestNextCell:
    def test_finds_up_cell(self):
        env = GridWorldEnv()
        env.agent_pos = (2, 2)
        cell, effect = env.next_cell(UP)
        assert cell == (1, 2)
        assert effect == MOVEMENT

    def test_finds_down_cell(self):
        env = GridWorldEnv()
        env.agent_pos = (2, 2)
        cell, effect = env.next_cell(DOWN)
        assert cell == (3, 2)
        assert effect == MOVEMENT

    def test_finds_left_cell(self):
        env = GridWorldEnv()
        env.agent_pos = (2, 2)
        cell, effect = env.next_cell(LEFT)
        assert cell == (2, 1)
        assert effect == MOVEMENT

    def test_finds_right_cell(self):
        env = GridWorldEnv()
        env.agent_pos = (2, 2)
        cell, effect = env.next_cell(RIGHT)
        assert cell == (2, 3)
        assert effect == MOVEMENT

    def test_finds_up_cell_when_off_board(self):
        env = GridWorldEnv()
        env.agent_pos = (0, 0)
        cell, effect = env.next_cell(UP)
        assert cell == (0, 0)
        assert effect == OFF_BOARD

    def test_finds_down_cell_when_off_board(self):
        env = GridWorldEnv()
        env.agent_pos = (4, 4)
        cell, effect = env.next_cell(DOWN)
        assert cell == (4, 4)
        assert effect == OFF_BOARD

    def test_finds_left_cell_when_off_board(self):
        env = GridWorldEnv()
        env.agent_pos = (0, 0)
        cell, effect = env.next_cell(LEFT)
        assert cell == (0, 0)
        assert effect == OFF_BOARD

    def test_finds_right_cell_when_off_board(self):
        env = GridWorldEnv()
        env.agent_pos = (4, 4)
        cell, effect = env.next_cell(RIGHT)
        assert cell == (4, 4)
        assert effect == OFF_BOARD

    def test_goes_to_goal(self):
        env = GridWorldEnv()
        env.agent_pos = (3, 4)
        cell, effect = env.next_cell(DOWN)
        assert cell == (4, 4)
        assert effect == GOAL

    def test_handles_obstacle(self):
        env = GridWorldEnv()
        env.agent_pos = (2, 2)
        env.get_cell((2, 1)).obstacle = True
        cell, effect = env.next_cell(LEFT)
        assert cell == (2, 2)
        assert effect == OBSTACLE
