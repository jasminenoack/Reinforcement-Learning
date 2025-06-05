import pytest
from tic_tac_logic.agents.mask_agent import (
    MaskHorizontal3,
    MaskHorizontal3X,
    MaskHorizontal3O,
    MaskAgent,
    MaskKey,
    MaskResult,
    ConfidentMask,
)
from tic_tac_logic.constants import X, O, E
from tic_tac_logic.sample_grids import get_easy_grid
from tic_tac_logic.env.grid import Grid


class TestMaskHorizontal3:
    @pytest.mark.parametrize(
        "coord, expected",
        [
            ((0, 0), None),
            (
                (0, 1),
                MaskKey(
                    mask_type=MaskHorizontal3,
                    pattern="XO_",
                    symbol=X,
                ),
            ),
            ((0, 2), None),
            ((1, 0), None),
            (
                (1, 1),
                MaskKey(
                    mask_type=MaskHorizontal3,
                    pattern="_XO",
                    symbol=X,
                ),
            ),
            ((1, 2), None),
            ((2, 0), None),
            (
                (2, 1),
                MaskKey(
                    mask_type=MaskHorizontal3,
                    pattern="O_X",
                    symbol=X,
                ),
            ),
            ((2, 2), None),
        ],
    )
    def test_get_mask(self, coord: tuple[int, int], expected: str | None):
        mask = MaskHorizontal3()
        result = mask.get_mask(coord, grid=[[X, O, E], [E, X, O], [O, E, X]], current=X)
        assert result == expected


class TestMaskHorizontal3X:
    @pytest.mark.parametrize(
        "coord, expected",
        [
            ((0, 0), None),
            (
                (0, 1),
                MaskKey(
                    mask_type=MaskHorizontal3X,
                    pattern="X__",
                    symbol=X,
                ),
            ),
            ((0, 2), None),
            ((1, 0), None),
            (
                (1, 1),
                MaskKey(
                    mask_type=MaskHorizontal3X,
                    pattern="_X_",
                    symbol=X,
                ),
            ),
            ((1, 2), None),
            ((2, 0), None),
            (
                (2, 1),
                MaskKey(
                    mask_type=MaskHorizontal3X,
                    pattern="__X",
                    symbol=X,
                ),
            ),
            ((2, 2), None),
        ],
    )
    def test_get_mask(self, coord: tuple[int, int], expected: str | None):
        mask = MaskHorizontal3X()
        result = mask.get_mask(coord, grid=[[X, O, E], [E, X, O], [O, E, X]], current=X)
        assert result == expected


class TestMaskHorizontal3O:
    @pytest.mark.parametrize(
        "coord, expected",
        [
            ((0, 0), None),
            (
                (0, 1),
                MaskKey(
                    mask_type=MaskHorizontal3O,
                    pattern="_O_",
                    symbol=O,
                ),
            ),
            ((0, 2), None),
            ((1, 0), None),
            (
                (1, 1),
                MaskKey(
                    mask_type=MaskHorizontal3O,
                    pattern="__O",
                    symbol=O,
                ),
            ),
            ((1, 2), None),
            ((2, 0), None),
            (
                (2, 1),
                MaskKey(
                    mask_type=MaskHorizontal3O,
                    pattern="O__",
                    symbol=O,
                ),
            ),
            ((2, 2), None),
        ],
    )
    def test_get_mask(self, coord: tuple[int, int], expected: str | None):
        mask = MaskHorizontal3O()
        result = mask.get_mask(coord, grid=[[X, O, E], [E, X, O], [O, E, X]], current=O)
        assert result == expected


class TestLearn:
    def test_learns_from_changes_1(self):
        easy_grid = get_easy_grid()
        # [     0  1  2  3  4  5
        # 0    [E, E, E, E, E, E],
        # 1    [X, X, E, E, O, E],
        # 2    [E, E, O, E, E, O],
        # 3    [E, X, E, E, E, E],
        # 4    [E, E, E, E, O, E],
        # 5    [X, E, X, E, E, E],
        # 6    [X, E, O, E, E, X],
        # 7    [E, E, E, E, E, E],
        # ]
        grid = Grid(easy_grid)
        agent = MaskAgent(easy_grid)

        # [     0  1  2  3  4  5
        # 0    [E, E, E, E, E, E],
        # 1    [X, X, O, E, O, E],
        # 2    [E, E, O, E, E, O],
        # 3    [E, X, E, E, E, E],
        # 4    [E, E, E, E, O, E],
        # 5    [X, E, X, E, E, E],
        # 6    [X, E, O, E, E, X],
        # 7    [E, E, E, E, E, E],
        # ]
        step = grid.act((1, 2), O)
        agent.learn(step)
        assert agent.q_table["masks"] == {
            MaskKey(
                mask_type=MaskHorizontal3,
                pattern="X__",
                symbol="O",
            ): MaskResult(
                mask=MaskKey(
                    mask_type=MaskHorizontal3,
                    pattern="X__",
                    symbol="O",
                ),
                failure_count=0,
                success_count=1,
            ),
            MaskKey(
                mask_type=MaskHorizontal3O,
                pattern="___",
                symbol="O",
            ): MaskResult(
                mask=MaskKey(
                    mask_type=MaskHorizontal3O,
                    pattern="___",
                    symbol="O",
                ),
                failure_count=0,
                success_count=1,
            ),
            MaskKey(
                mask_type=MaskHorizontal3X,
                pattern="X__",
                symbol="O",
            ): MaskResult(
                mask=MaskKey(
                    mask_type=MaskHorizontal3X,
                    pattern="X__",
                    symbol="O",
                ),
                failure_count=0,
                success_count=1,
            ),
        }

        # [     0  1  2  3  4  5
        # 0    [E, E, X, E, E, E],
        # 1    [X, X, O, E, O, E],
        # 2    [E, E, O, E, E, O],
        # 3    [E, X, E, E, E, E],
        # 4    [E, E, E, E, O, E],
        # 5    [X, E, X, E, E, E],
        # 6    [X, E, O, E, E, X],
        # 7    [E, E, E, E, E, E],
        # ]
        step = grid.act((0, 2), X)
        agent.learn(step)
        assert agent.q_table["masks"] == {
            MaskKey(
                mask_type=MaskHorizontal3,
                pattern="X__",
                symbol="O",
            ): MaskResult(
                mask=MaskKey(
                    mask_type=MaskHorizontal3,
                    pattern="X__",
                    symbol="O",
                ),
                failure_count=0,
                success_count=1,
            ),
            MaskKey(
                mask_type=MaskHorizontal3O,
                pattern="___",
                symbol="O",
            ): MaskResult(
                mask=MaskKey(
                    mask_type=MaskHorizontal3O,
                    pattern="___",
                    symbol="O",
                ),
                failure_count=0,
                success_count=1,
            ),
            MaskKey(
                mask_type=MaskHorizontal3X,
                pattern="X__",
                symbol="O",
            ): MaskResult(
                mask=MaskKey(
                    mask_type=MaskHorizontal3X,
                    pattern="X__",
                    symbol="O",
                ),
                failure_count=0,
                success_count=1,
            ),
            MaskKey(
                mask_type=MaskHorizontal3,
                pattern="___",
                symbol="X",
            ): MaskResult(
                mask=MaskKey(
                    mask_type=MaskHorizontal3,
                    pattern="___",
                    symbol="X",
                ),
                failure_count=0,
                success_count=1,
            ),
            MaskKey(
                mask_type=MaskHorizontal3O,
                pattern="___",
                symbol="X",
            ): MaskResult(
                mask=MaskKey(
                    mask_type=MaskHorizontal3O,
                    pattern="___",
                    symbol="X",
                ),
                failure_count=0,
                success_count=1,
            ),
            MaskKey(
                mask_type=MaskHorizontal3X,
                pattern="___",
                symbol="X",
            ): MaskResult(
                mask=MaskKey(
                    mask_type=MaskHorizontal3X,
                    pattern="___",
                    symbol="X",
                ),
                failure_count=0,
                success_count=1,
            ),
        }

        # [     0  1  2  3  4  5
        # 0    [E, E, X, E, E, E],
        # 1    [X, X, O, O, O, E],
        # 2    [E, E, O, E, E, O],
        # 3    [E, X, E, E, E, E],
        # 4    [E, E, E, E, O, E],
        # 5    [X, E, X, E, E, E],
        # 6    [X, E, O, E, E, X],
        # 7    [E, E, E, E, E, E],
        # ]
        step = grid.act((1, 3), O)
        agent.learn(step)
        assert agent.q_table["masks"] == {
            MaskKey(
                mask_type=MaskHorizontal3O,
                pattern="___",
                symbol="O",
            ): MaskResult(
                mask=MaskKey(
                    mask_type=MaskHorizontal3O,
                    pattern="___",
                    symbol="O",
                ),
                failure_count=0,
                success_count=1,
            ),
            MaskKey(
                mask_type=MaskHorizontal3X,
                pattern="X__",
                symbol="O",
            ): MaskResult(
                mask=MaskKey(
                    mask_type=MaskHorizontal3X,
                    pattern="X__",
                    symbol="O",
                ),
                failure_count=0,
                success_count=1,
            ),
            MaskKey(
                mask_type=MaskHorizontal3,
                pattern="X__",
                symbol="O",
            ): MaskResult(
                mask=MaskKey(
                    mask_type=MaskHorizontal3,
                    pattern="X__",
                    symbol="O",
                ),
                failure_count=0,
                success_count=1,
            ),
            MaskKey(
                mask_type=MaskHorizontal3O,
                pattern="___",
                symbol="X",
            ): MaskResult(
                mask=MaskKey(
                    mask_type=MaskHorizontal3O,
                    pattern="___",
                    symbol="X",
                ),
                failure_count=0,
                success_count=1,
            ),
            MaskKey(
                mask_type=MaskHorizontal3X,
                pattern="___",
                symbol="X",
            ): MaskResult(
                mask=MaskKey(
                    mask_type=MaskHorizontal3X,
                    pattern="___",
                    symbol="X",
                ),
                failure_count=0,
                success_count=1,
            ),
            MaskKey(
                mask_type=MaskHorizontal3,
                pattern="___",
                symbol="X",
            ): MaskResult(
                mask=MaskKey(
                    mask_type=MaskHorizontal3,
                    pattern="___",
                    symbol="X",
                ),
                failure_count=0,
                success_count=1,
            ),
            MaskKey(
                mask_type=MaskHorizontal3,
                pattern="O_O",
                symbol="O",
            ): MaskResult(
                mask=MaskKey(
                    mask_type=MaskHorizontal3,
                    pattern="O_O",
                    symbol="O",
                ),
                failure_count=1,
                success_count=0,
            ),
            MaskKey(
                mask_type=MaskHorizontal3O,
                pattern="O_O",
                symbol="O",
            ): MaskResult(
                mask=MaskKey(
                    mask_type=MaskHorizontal3O,
                    pattern="O_O",
                    symbol="O",
                ),
                failure_count=1,
                success_count=0,
            ),
            MaskKey(
                mask_type=MaskHorizontal3X,
                pattern="___",
                symbol="O",
            ): MaskResult(
                mask=MaskKey(
                    mask_type=MaskHorizontal3X,
                    pattern="___",
                    symbol="O",
                ),
                failure_count=1,
                success_count=0,
            ),
        }

    def test_learns_from_changes_2(self):
        easy_grid = get_easy_grid()
        # [     0  1  2  3  4  5
        # 0    [E, E, E, E, E, E],
        # 1    [X, X, E, E, O, E],
        # 2    [E, E, O, E, E, O],
        # 3    [E, X, E, E, E, E],
        # 4    [E, E, E, E, O, E],
        # 5    [X, E, X, E, E, E],
        # 6    [X, E, O, E, E, X],
        # 7    [E, E, E, E, E, E],
        # ]
        grid = Grid(easy_grid)
        agent = MaskAgent(easy_grid)

        # 1, 3  - E, E, O - X
        step = grid.act((1, 3), X)
        agent.learn(step)
        # 1, 2 - X E X - O
        step = grid.act((1, 2), O)
        agent.learn(step)
        # 5, 1 - X E X - O
        step = grid.act((5, 1), O)
        agent.learn(step)

        assert agent.q_table["masks"] == {
            MaskKey(
                mask_type=MaskHorizontal3,
                pattern="__O",
                symbol="X",
            ): MaskResult(
                mask=MaskKey(
                    mask_type=MaskHorizontal3,
                    pattern="__O",
                    symbol="X",
                ),
                failure_count=0,
                success_count=1,
            ),
            MaskKey(
                mask_type=MaskHorizontal3O,
                pattern="__O",
                symbol="X",
            ): MaskResult(
                mask=MaskKey(
                    mask_type=MaskHorizontal3O,
                    pattern="__O",
                    symbol="X",
                ),
                failure_count=0,
                success_count=1,
            ),
            MaskKey(
                mask_type=MaskHorizontal3X,
                pattern="___",
                symbol="X",
            ): MaskResult(
                mask=MaskKey(
                    mask_type=MaskHorizontal3X,
                    pattern="___",
                    symbol="X",
                ),
                failure_count=0,
                success_count=1,
            ),
            MaskKey(
                mask_type=MaskHorizontal3,
                pattern="X_X",
                symbol="O",
            ): MaskResult(
                mask=MaskKey(
                    mask_type=MaskHorizontal3,
                    pattern="X_X",
                    symbol="O",
                ),
                failure_count=0,
                success_count=2,
            ),
            MaskKey(
                mask_type=MaskHorizontal3O,
                pattern="___",
                symbol="O",
            ): MaskResult(
                mask=MaskKey(
                    mask_type=MaskHorizontal3O,
                    pattern="___",
                    symbol="O",
                ),
                failure_count=0,
                success_count=2,
            ),
            MaskKey(
                mask_type=MaskHorizontal3X,
                pattern="X_X",
                symbol="O",
            ): MaskResult(
                mask=MaskKey(
                    mask_type=MaskHorizontal3X,
                    pattern="X_X",
                    symbol="O",
                ),
                failure_count=0,
                success_count=2,
            ),
        }

    # [     0  1  2  3  4  5
    # 0    [E, E, E, E, E, E],
    # 1    [X, X, E, E, O, E],
    # 2    [E, E, O, E, E, O],
    # 3    [E, X, E, E, E, E],
    # 4    [E, E, E, E, O, E],
    # 5    [X, E, X, E, E, E],
    # 6    [X, E, O, E, E, X],
    # 7    [E, E, E, E, E, E],
    # ]


class TestFindAgressiveFailures:

    def test_finds_cases_where_we_are_confident_an_answer_is_incorrect(self):
        easy_grid = get_easy_grid()
        agent = MaskAgent(easy_grid)

        agent.q_table["masks"] = {
            MaskKey(
                mask_type=MaskHorizontal3,
                pattern="X__",
                symbol="O",
            ): MaskResult(
                mask=MaskKey(
                    mask_type=MaskHorizontal3,
                    pattern="X__",
                    symbol="O",
                ),
                failure_count=7,
                success_count=1,
            ),
            MaskKey(
                mask_type=MaskHorizontal3O,
                pattern="___",
                symbol="O",
            ): MaskResult(
                mask=MaskKey(
                    mask_type=MaskHorizontal3O,
                    pattern="___",
                    symbol="O",
                ),
                failure_count=4,
                success_count=0,
            ),
            MaskKey(
                mask_type=MaskHorizontal3X,
                pattern="X__",
                symbol="O",
            ): MaskResult(
                mask=MaskKey(
                    mask_type=MaskHorizontal3X,
                    pattern="X__",
                    symbol="O",
                ),
                failure_count=100,
                success_count=1,
            ),
            MaskKey(
                mask_type=MaskHorizontal3,
                pattern="O_O",
                symbol="X",
            ): MaskResult(
                mask=MaskKey(
                    mask_type=MaskHorizontal3,
                    pattern="O_O",
                    symbol="X",
                ),
                failure_count=5,
                success_count=0,
            ),
            MaskKey(
                mask_type=MaskHorizontal3O,
                pattern="O_O",
                symbol="O",
            ): MaskResult(
                mask=MaskKey(
                    mask_type=MaskHorizontal3O,
                    pattern="O_O",
                    symbol="O",
                ),
                failure_count=10,
                success_count=0,
            ),
            MaskKey(
                mask_type=MaskHorizontal3X,
                pattern="___",
                symbol="O",
            ): MaskResult(
                mask=MaskKey(
                    mask_type=MaskHorizontal3X,
                    pattern="___",
                    symbol="O",
                ),
                failure_count=1,
                success_count=0,
            ),
        }

        assert agent.find_aggressive_failures() == {
            ConfidentMask(
                mask_key=MaskKey(
                    mask_type=MaskHorizontal3,
                    pattern="O_O",
                    symbol="X",
                ),
                prediction=-1.0,
            ),
            ConfidentMask(
                mask_key=MaskKey(
                    mask_type=MaskHorizontal3O,
                    pattern="O_O",
                    symbol="O",
                ),
                prediction=-1.0,
            ),
        }


class TestRemovePossibleMoves:
    def test_removes_failing_options(self):
        easy_grid = get_easy_grid()
        # [     0  1  2  3  4  5
        # 0    [E, E, E, E, E, E],
        # 1    [X, X, E, E, O, E],
        # 2    [E, E, O, E, E, O],
        # 3    [E, X, E, E, E, E],
        # 4    [E, E, E, E, O, E],
        # 5    [X, E, X, E, E, E],
        # 6    [X, E, O, E, E, X],
        # 7    [E, E, E, E, E, E],
        # ]
        agent = MaskAgent(easy_grid)

        possible_moves = {
            ((0, 0), X),
            ((0, 1), O),
            ((0, 1), X),
            ((1, 2), X),
            ((1, 2), O),
            ((1, 3), O),
        }

        failure_masks = {
            ConfidentMask(
                mask_key=MaskKey(
                    mask_type=MaskHorizontal3,
                    pattern="X__",
                    symbol="O",
                ),
                prediction=-1.0,
            ),
            ConfidentMask(
                mask_key=MaskKey(
                    mask_type=MaskHorizontal3O,
                    pattern="___",
                    symbol="O",
                ),
                prediction=-1.0,
            ),
        }
        assert agent.remove_failing_options(
            easy_grid, possible_moves, failure_masks
        ) == {
            ((0, 0), X),
            ((0, 1), X),
            ((1, 2), X),
            ((1, 3), O),
        }


class TestOptionsWithOneChoice:
    def test_returns_options_with_one_choice(self):
        easy_grid = get_easy_grid()
        agent = MaskAgent(easy_grid)

        possible_moves = {
            ((0, 0), X),
            ((0, 1), O),
            ((0, 1), X),
            ((1, 2), X),
            ((1, 2), O),
            ((1, 3), O),
        }
        options = agent.options_with_one_choice(possible_moves)

        assert options == {
            ((0, 0), X),
            ((1, 3), O),
        }
