import pytest
import random
from tic_tac_logic.agents.mask_agent import (
    generate_pool_masks,
    MaskAgent,
    MaskResult,
)
from tic_tac_logic.constants import X, O, E, Observation
from tic_tac_logic.sample_grids import get_easy_grid
from tic_tac_logic.env.grid import Grid
from tic_tac_logic.agents.masks import (
    MaskRules,
    CompleteMask,
)

masks = list(generate_pool_masks(3, 3))
MaskHorizontal3Centered = [mask for mask in masks if mask.name == "Mask|1x3|(0, 1)|X"][
    0
]
MaskHorizontal3CenteredX = [
    mask for mask in masks if mask.name == "Mask|1x3|(0, 1)|<X>|X"
][0]
MaskHorizontal3CenteredO = [
    mask for mask in masks if mask.name == "Mask|1x3|(0, 1)|<O>|X"
][0]


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
        agent = MaskAgent(len(easy_grid), len(easy_grid[0]))

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

        for mask in agent.q_table["masks"]:
            print(mask)

        for mask, mask_result in {
            CompleteMask(
                pattern="X__",
                match_symbol="X",
                rules=MaskRules(
                    rows_above=0,
                    rows_below=0,
                    columns_left=1,
                    columns_right=1,
                ),
                symbol_to_place=O,
            ): MaskResult(
                mask=CompleteMask(
                    pattern="X__",
                    match_symbol="X",
                    rules=MaskRules(
                        rows_above=0,
                        rows_below=0,
                        columns_left=1,
                        columns_right=1,
                    ),
                    symbol_to_place=O,
                ),
                failure_count=0,
                success_count=1,
            ),
            CompleteMask(
                pattern="___",
                match_symbol="O",
                rules=MaskRules(
                    rows_above=0,
                    rows_below=0,
                    columns_left=1,
                    columns_right=1,
                ),
                symbol_to_place=O,
            ): MaskResult(
                mask=CompleteMask(
                    pattern="___",
                    match_symbol="O",
                    rules=MaskRules(
                        rows_above=0,
                        rows_below=0,
                        columns_left=1,
                        columns_right=1,
                    ),
                    symbol_to_place=O,
                ),
                failure_count=0,
                success_count=1,
            ),
            CompleteMask(
                pattern="X__",
                match_symbol="X",
                rules=MaskRules(
                    rows_above=0,
                    rows_below=0,
                    columns_left=1,
                    columns_right=1,
                ),
                symbol_to_place=O,
            ): MaskResult(
                mask=CompleteMask(
                    pattern="X__",
                    match_symbol="X",
                    rules=MaskRules(
                        rows_above=0,
                        rows_below=0,
                        columns_left=1,
                        columns_right=1,
                    ),
                    symbol_to_place=O,
                ),
                failure_count=0,
                success_count=1,
            ),
        }.items():
            assert agent.q_table["masks"][mask] == mask_result

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
        for mask, mask_result in {
            CompleteMask(
                pattern="X__",
                symbol_to_place="O",
                match_symbol="X",
                rules=MaskRules(
                    rows_above=0,
                    rows_below=0,
                    columns_left=1,
                    columns_right=1,
                ),
            ): MaskResult(
                mask=CompleteMask(
                    pattern="X__",
                    symbol_to_place="O",
                    match_symbol="X",
                    rules=MaskRules(
                        rows_above=0,
                        rows_below=0,
                        columns_left=1,
                        columns_right=1,
                    ),
                ),
                failure_count=0,
                success_count=1,
            ),
            CompleteMask(
                pattern="___",
                symbol_to_place="O",
                match_symbol="O",
                rules=MaskRules(
                    rows_above=0,
                    rows_below=0,
                    columns_left=1,
                    columns_right=1,
                ),
            ): MaskResult(
                mask=CompleteMask(
                    pattern="___",
                    symbol_to_place="O",
                    match_symbol="O",
                    rules=MaskRules(
                        rows_above=0,
                        rows_below=0,
                        columns_left=1,
                        columns_right=1,
                    ),
                ),
                failure_count=0,
                success_count=1,
            ),
            CompleteMask(
                pattern="X__",
                symbol_to_place="O",
                match_symbol=None,
                rules=MaskRules(
                    rows_above=0,
                    rows_below=0,
                    columns_left=1,
                    columns_right=1,
                ),
            ): MaskResult(
                mask=CompleteMask(
                    match_symbol=None,
                    pattern="X__",
                    symbol_to_place="O",
                    rules=MaskRules(
                        rows_above=0,
                        rows_below=0,
                        columns_left=1,
                        columns_right=1,
                    ),
                ),
                failure_count=0,
                success_count=1,
            ),
            CompleteMask(
                pattern="___",
                symbol_to_place="X",
                match_symbol=None,
                rules=MaskRules(
                    rows_above=0,
                    rows_below=0,
                    columns_left=1,
                    columns_right=1,
                ),
            ): MaskResult(
                mask=CompleteMask(
                    match_symbol=None,
                    pattern="___",
                    symbol_to_place="X",
                    rules=MaskRules(
                        rows_above=0,
                        rows_below=0,
                        columns_left=1,
                        columns_right=1,
                    ),
                ),
                failure_count=0,
                success_count=1,
            ),
            CompleteMask(
                pattern="___",
                symbol_to_place="X",
                match_symbol="X",
                rules=MaskRules(
                    rows_above=0,
                    rows_below=0,
                    columns_left=1,
                    columns_right=1,
                ),
            ): MaskResult(
                mask=CompleteMask(
                    match_symbol="X",
                    pattern="___",
                    symbol_to_place="X",
                    rules=MaskRules(
                        rows_above=0,
                        rows_below=0,
                        columns_left=1,
                        columns_right=1,
                    ),
                ),
                failure_count=0,
                success_count=1,
            ),
            CompleteMask(
                pattern="___",
                symbol_to_place="X",
                match_symbol="O",
                rules=MaskRules(
                    rows_above=0,
                    rows_below=0,
                    columns_left=1,
                    columns_right=1,
                ),
            ): MaskResult(
                mask=CompleteMask(
                    match_symbol="O",
                    pattern="___",
                    symbol_to_place="X",
                    rules=MaskRules(
                        rows_above=0,
                        rows_below=0,
                        columns_left=1,
                        columns_right=1,
                    ),
                ),
                failure_count=0,
                success_count=1,
            ),
        }.items():
            assert agent.q_table["masks"][mask] == mask_result

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
        agent = MaskAgent(len(easy_grid), len(easy_grid[0]))

        # 1, 3  - E, E, O - X
        step = grid.act((1, 3), X)
        agent.learn(step)
        # 1, 2 - X E X - O
        step = grid.act((1, 2), O)
        agent.learn(step)
        # 5, 1 - X E X - O
        step = grid.act((5, 1), O)
        agent.learn(step)

        for mask, mask_result in {
            CompleteMask(
                pattern="__O",
                symbol_to_place="X",
                match_symbol=None,
                rules=MaskRules(
                    rows_above=0,
                    rows_below=0,
                    columns_left=1,
                    columns_right=1,
                ),
            ): MaskResult(
                mask=CompleteMask(
                    rules=MaskRules(
                        rows_above=0,
                        rows_below=0,
                        columns_left=1,
                        columns_right=1,
                    ),
                    pattern="__O",
                    symbol_to_place="X",
                    match_symbol=None,
                ),
                failure_count=0,
                success_count=1,
            ),
            CompleteMask(
                rules=MaskRules(
                    rows_above=0,
                    rows_below=0,
                    columns_left=1,
                    columns_right=1,
                ),
                match_symbol="O",
                pattern="__O",
                symbol_to_place="X",
            ): MaskResult(
                mask=CompleteMask(
                    rules=MaskRules(
                        rows_above=0,
                        rows_below=0,
                        columns_left=1,
                        columns_right=1,
                    ),
                    match_symbol="O",
                    pattern="__O",
                    symbol_to_place="X",
                ),
                failure_count=0,
                success_count=1,
            ),
            CompleteMask(
                rules=MaskRules(
                    rows_above=0,
                    rows_below=0,
                    columns_left=1,
                    columns_right=1,
                ),
                match_symbol="X",
                pattern="___",
                symbol_to_place="X",
            ): MaskResult(
                mask=CompleteMask(
                    rules=MaskRules(
                        rows_above=0,
                        rows_below=0,
                        columns_left=1,
                        columns_right=1,
                    ),
                    match_symbol="X",
                    pattern="___",
                    symbol_to_place="X",
                ),
                failure_count=0,
                success_count=1,
            ),
            CompleteMask(
                rules=MaskRules(
                    rows_above=0,
                    rows_below=0,
                    columns_left=1,
                    columns_right=1,
                ),
                pattern="X_X",
                symbol_to_place="O",
                match_symbol=None,
            ): MaskResult(
                mask=CompleteMask(
                    rules=MaskRules(
                        rows_above=0,
                        rows_below=0,
                        columns_left=1,
                        columns_right=1,
                    ),
                    pattern="X_X",
                    symbol_to_place="O",
                    match_symbol=None,
                ),
                failure_count=0,
                success_count=2,
            ),
            CompleteMask(
                rules=MaskRules(
                    rows_above=0,
                    rows_below=0,
                    columns_left=1,
                    columns_right=1,
                ),
                match_symbol="O",
                pattern="___",
                symbol_to_place="O",
            ): MaskResult(
                mask=CompleteMask(
                    rules=MaskRules(
                        rows_above=0,
                        rows_below=0,
                        columns_left=1,
                        columns_right=1,
                    ),
                    match_symbol="O",
                    pattern="___",
                    symbol_to_place="O",
                ),
                failure_count=0,
                success_count=2,
            ),
            CompleteMask(
                rules=MaskRules(
                    rows_above=0,
                    rows_below=0,
                    columns_left=1,
                    columns_right=1,
                ),
                match_symbol="X",
                pattern="X_X",
                symbol_to_place="O",
            ): MaskResult(
                mask=CompleteMask(
                    rules=MaskRules(
                        rows_above=0,
                        rows_below=0,
                        columns_left=1,
                        columns_right=1,
                    ),
                    match_symbol="X",
                    pattern="X_X",
                    symbol_to_place="O",
                ),
                failure_count=0,
                success_count=2,
            ),
        }.items():
            assert agent.q_table["masks"][mask] == mask_result

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
        agent = MaskAgent(len(easy_grid), len(easy_grid[0]))

        agent.q_table["masks"] = {
            CompleteMask(
                rules=MaskRules(
                    rows_above=0,
                    rows_below=0,
                    columns_left=1,
                    columns_right=1,
                ),
                match_symbol=None,
                pattern="X__",
                symbol_to_place="O",
            ): MaskResult(
                mask=CompleteMask(
                    rules=MaskRules(
                        rows_above=0,
                        rows_below=0,
                        columns_left=1,
                        columns_right=1,
                    ),
                    pattern="X__",
                    match_symbol=None,
                    symbol_to_place="O",
                ),
                failure_count=7,
                success_count=1,
            ),
            CompleteMask(
                rules=MaskRules(
                    rows_above=0,
                    rows_below=0,
                    columns_left=1,
                    columns_right=1,
                ),
                match_symbol="O",
                pattern="___",
                symbol_to_place="O",
            ): MaskResult(
                mask=CompleteMask(
                    rules=MaskRules(
                        rows_above=0,
                        rows_below=0,
                        columns_left=1,
                        columns_right=1,
                    ),
                    match_symbol="O",
                    pattern="___",
                    symbol_to_place="O",
                ),
                failure_count=4,
                success_count=0,
            ),
            CompleteMask(
                rules=MaskRules(
                    rows_above=0,
                    rows_below=0,
                    columns_left=1,
                    columns_right=1,
                ),
                match_symbol="X",
                pattern="X__",
                symbol_to_place="O",
            ): MaskResult(
                mask=CompleteMask(
                    rules=MaskRules(
                        rows_above=0,
                        rows_below=0,
                        columns_left=1,
                        columns_right=1,
                    ),
                    match_symbol="X",
                    pattern="X__",
                    symbol_to_place="O",
                ),
                failure_count=100,
                success_count=1,
            ),
            CompleteMask(
                rules=MaskRules(
                    rows_above=0,
                    rows_below=0,
                    columns_left=1,
                    columns_right=1,
                ),
                pattern="O_O",
                match_symbol=None,
                symbol_to_place="X",
            ): MaskResult(
                mask=CompleteMask(
                    rules=MaskRules(
                        rows_above=0,
                        rows_below=0,
                        columns_left=1,
                        columns_right=1,
                    ),
                    pattern="O_O",
                    match_symbol=None,
                    symbol_to_place="X",
                ),
                failure_count=5,
                success_count=0,
            ),
            CompleteMask(
                rules=MaskRules(
                    rows_above=0,
                    rows_below=0,
                    columns_left=1,
                    columns_right=1,
                ),
                match_symbol="O",
                pattern="O_O",
                symbol_to_place="O",
            ): MaskResult(
                mask=CompleteMask(
                    rules=MaskRules(
                        rows_above=0,
                        rows_below=0,
                        columns_left=1,
                        columns_right=1,
                    ),
                    match_symbol="O",
                    pattern="O_O",
                    symbol_to_place="O",
                ),
                failure_count=10,
                success_count=0,
            ),
            CompleteMask(
                rules=MaskRules(
                    rows_above=0,
                    rows_below=0,
                    columns_left=1,
                    columns_right=1,
                ),
                match_symbol="X",
                pattern="___",
                symbol_to_place="O",
            ): MaskResult(
                mask=CompleteMask(
                    rules=MaskRules(
                        rows_above=0,
                        rows_below=0,
                        columns_left=1,
                        columns_right=1,
                    ),
                    match_symbol="X",
                    pattern="___",
                    symbol_to_place="O",
                ),
                failure_count=1,
                success_count=0,
            ),
        }

        assert agent.find_aggressive_failures() == {
            CompleteMask(
                rules=MaskRules(
                    rows_above=0,
                    rows_below=0,
                    columns_left=1,
                    columns_right=1,
                ),
                pattern="O_O",
                symbol_to_place="X",
                match_symbol=None,
            ),
            CompleteMask(
                rules=MaskRules(
                    rows_above=0,
                    rows_below=0,
                    columns_left=1,
                    columns_right=1,
                ),
                match_symbol="O",
                pattern="O_O",
                symbol_to_place="O",
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
        agent = MaskAgent(len(easy_grid), len(easy_grid[0]))

        possible_moves = {
            ((0, 0), X),
            ((0, 1), O),
            ((0, 1), X),
            ((1, 2), X),
            ((1, 2), O),
            ((1, 3), O),
        }

        failure_masks = {
            CompleteMask(
                rules=MaskRules(
                    rows_above=0,
                    rows_below=0,
                    columns_left=1,
                    columns_right=1,
                ),
                pattern="X__",
                symbol_to_place="O",
                match_symbol="X",
            ),
            CompleteMask(
                rules=MaskRules(
                    rows_above=0,
                    rows_below=0,
                    columns_left=1,
                    columns_right=1,
                ),
                match_symbol="O",
                pattern="___",
                symbol_to_place="O",
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

    @pytest.mark.parametrize(
        "possible_moves,failure_masks,expected",
        [  # pyright: ignore
            [
                {((0, 0), X)},
                set(),
                {((0, 0), X)},
            ],
            [
                {
                    ((0, 0), X),
                    ((0, 1), O),
                    ((0, 1), X),
                    ((1, 2), X),
                    ((1, 2), O),
                    ((1, 3), O),
                },
                {
                    CompleteMask(
                        rules=MaskRules(
                            rows_above=0,
                            rows_below=0,
                            columns_left=1,
                            columns_right=1,
                        ),
                        pattern="X__",
                        match_symbol=None,
                        symbol_to_place="O",
                    ),
                    CompleteMask(
                        rules=MaskRules(
                            rows_above=0,
                            rows_below=0,
                            columns_left=1,
                            columns_right=1,
                        ),
                        match_symbol="O",
                        pattern="___",
                        symbol_to_place="O",
                    ),
                },
                {
                    ((0, 0), X),
                    ((0, 1), X),
                    ((1, 2), X),
                    ((1, 3), O),
                },
            ],
            [
                {((1, 2), O), ((1, 2), X)},
                {
                    CompleteMask(
                        rules=MaskRules(
                            rows_above=0,
                            rows_below=0,
                            columns_left=1,
                            columns_right=1,
                        ),
                        pattern="X__",
                        match_symbol=None,
                        symbol_to_place="O",
                    ),
                    CompleteMask(
                        rules=MaskRules(
                            rows_above=0,
                            rows_below=0,
                            columns_left=1,
                            columns_right=1,
                        ),
                        pattern="X__",
                        match_symbol=None,
                        symbol_to_place="X",
                    ),
                },
                set(),
            ],
            [
                {((9, 0), X), ((0, 0), X)},
                {
                    CompleteMask(
                        rules=MaskRules(
                            rows_above=0,
                            rows_below=0,
                            columns_left=1,
                            columns_right=1,
                        ),
                        pattern="X__",
                        symbol_to_place="X",
                        match_symbol=None,
                    ),
                },
                {((9, 0), X), ((0, 0), X)},
            ],
        ],
    )
    def test_remove_failing_options_parametrized(
        self,
        possible_moves: set[tuple[tuple[int, int], str]],
        failure_masks: set[CompleteMask],
        expected: set[tuple[tuple[int, int], str]],
    ) -> None:
        grid = get_easy_grid()
        agent = MaskAgent(len(grid), len(grid[0]))
        assert (
            agent.remove_failing_options(grid, possible_moves, failure_masks)
            == expected
        )


class TestOptionsWithOneChoice:
    def test_returns_options_with_one_choice(self):
        easy_grid = get_easy_grid()
        agent = MaskAgent(len(easy_grid), len(easy_grid[0]))

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


class TestAct:
    def test_raises_if_no_empty_cells(self):
        grid = [[X, O], [O, X]]
        agent = MaskAgent(len(grid), len(grid[0]))
        with pytest.raises(ValueError):
            agent.act(Observation(grid))

    def test_returns_random_move_when_exploring(self):
        grid = [[E, E], [E, E]]
        agent = MaskAgent(2, 2)
        agent.epsilon = 1
        random.seed(0)
        result = agent.act(Observation(grid))
        assert result == ((1, 1), "X")

    def test_returns_best_option_when_available(self, mocker):
        grid = [[E, E], [E, E]]
        agent = MaskAgent(2, 2)
        mocker.patch.object(
            MaskAgent, "do_not_discover", autospec=True, return_value=True
        )
        mocker.patch.object(
            MaskAgent, "find_aggressive_failures", autospec=True, return_value=set()
        )
        mocker.patch.object(
            MaskAgent,
            "remove_failing_options",
            autospec=True,
            return_value={((0, 0), X)},
        )
        assert agent.act(Observation(grid)) == ((0, 0), X)

    def test_returns_random_when_no_best_option(self, mocker):
        grid = [[E, E], [E, E]]
        agent = MaskAgent(2, 2)
        random.seed(0)
        mocker.patch.object(
            MaskAgent, "do_not_discover", autospec=True, return_value=True
        )
        mocker.patch.object(
            MaskAgent, "find_aggressive_failures", autospec=True, return_value=set()
        )
        mocker.patch.object(
            MaskAgent,
            "remove_failing_options",
            autospec=True,
            side_effect=lambda self, g, pm, fm: pm,
        )
        mocker.patch.object(
            MaskAgent, "options_with_one_choice", autospec=True, return_value=set()
        )
        assert agent.act(Observation(grid)) == ((1, 1), "O")
