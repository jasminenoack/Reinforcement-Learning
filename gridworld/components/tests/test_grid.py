from hmac import new
import pytest
from ..grid_environment import GridWorldEnv


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
        assert reward == 99
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
