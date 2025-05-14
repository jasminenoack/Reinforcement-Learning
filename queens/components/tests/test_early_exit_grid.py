from queens.dtos import FailureType
from queens.utils import build_board_array
from queens.components.grid import EarlyExitGrid
import pytest


class TestStep:
    def test_fails_if_board_already_failed(self):
        grid = EarlyExitGrid(
            build_board_array(
                [
                    (0, 4),
                    (5, 4),
                ]
            )
        )
        with pytest.raises(ValueError):
            grid.step(*(2, 7))

    def test_returns_reward_for_step(self):
        grid = EarlyExitGrid(
            build_board_array(
                [
                    (0, 4),
                ]
            )
        )
        result = grid.step(*(2, 7))
        assert result.reward == 10
        assert result.failure_type is None
        assert result.board_state.board.tolist() == grid.board.tolist()

    def test_returns_reward_for_failure(self):
        grid = EarlyExitGrid(
            build_board_array(
                [
                    (0, 4),
                ]
            )
        )
        result = grid.step(*(5, 4))
        assert result.reward == -90
        assert result.failure_type == FailureType.COLUMN

    def test_returns_reward_for_diagonal_failure(self):
        grid = EarlyExitGrid(
            build_board_array(
                [
                    (0, 4),
                ]
            )
        )
        result = grid.step(*(1, 3))
        assert result.reward == -90
        assert result.failure_type == FailureType.REVERSE_DIAGONAL

    def test_returns_reward_for_reverse_diagonal_failure(self):
        grid = EarlyExitGrid(
            build_board_array(
                [
                    (0, 4),
                ]
            )
        )
        result = grid.step(*(1, 5))
        assert result.reward == -90
        assert result.failure_type == FailureType.DIAGONAL

    def test_returns_reward_for_two_in_row(self):
        grid = EarlyExitGrid(
            build_board_array(
                [
                    (0, 4),
                ]
            )
        )
        result = grid.step(*(0, 5))
        assert result.reward == -90
        assert result.failure_type == FailureType.ROW

    def test_returns_reward_for_success(self):
        grid = EarlyExitGrid(
            build_board_array(
                [
                    (0, 0),
                    (1, 4),
                    (2, 7),
                    (3, 5),
                    (4, 2),
                    (5, 6),
                    (6, 1),
                ]
            )
        )
        result = grid.step(*(7, 3))
        assert result.reward == 110


class TestDone:
    def test_done_if_failed(self):
        grid = EarlyExitGrid(
            build_board_array(
                [
                    (0, 4),
                    (5, 4),
                ]
            )
        )
        assert grid.done is True


class TestGetState:
    def test_returns_board_state(self):
        grid = EarlyExitGrid(
            build_board_array(
                [
                    (0, 4),
                    (5, 4),
                ]
            )
        )
        state = grid.get_state()
        assert state.board.tolist() == grid.board.tolist()
