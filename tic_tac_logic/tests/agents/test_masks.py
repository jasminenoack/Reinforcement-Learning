import pytest
from tic_tac_logic.agents.masks import (
    generate_pool_masks,
    MaskRules,
    AbstractMask,
    MaskKey,
)
from tic_tac_logic.constants import X, O, E


class TestGeneratePoolMasks:
    def test_generates_1x1_mask(self):
        masks = generate_pool_masks(1, 1)
        assert masks == [
            AbstractMask(
                match_symbol="X",
                rule=MaskRules(
                    rows_above=0, rows_below=0, columns_left=0, columns_right=0
                ),
            ),
            AbstractMask(
                match_symbol="O",
                rule=MaskRules(
                    rows_above=0, rows_below=0, columns_left=0, columns_right=0
                ),
            ),
            AbstractMask(
                match_symbol=None,
                rule=MaskRules(
                    rows_above=0, rows_below=0, columns_left=0, columns_right=0
                ),
            ),
        ]

    def test_generates_2x2_masks(self):
        masks = generate_pool_masks(2, 2)
        assert len(masks) == 27
        assert masks == [
            AbstractMask(
                match_symbol="X",
                rule=MaskRules(
                    rows_above=0, rows_below=0, columns_left=0, columns_right=0
                ),
            ),
            AbstractMask(
                match_symbol="O",
                rule=MaskRules(
                    rows_above=0, rows_below=0, columns_left=0, columns_right=0
                ),
            ),
            AbstractMask(
                match_symbol=None,
                rule=MaskRules(
                    rows_above=0, rows_below=0, columns_left=0, columns_right=0
                ),
            ),
            AbstractMask(
                match_symbol="X",
                rule=MaskRules(
                    rows_above=0, rows_below=0, columns_left=0, columns_right=1
                ),
            ),
            AbstractMask(
                match_symbol="O",
                rule=MaskRules(
                    rows_above=0, rows_below=0, columns_left=0, columns_right=1
                ),
            ),
            AbstractMask(
                match_symbol=None,
                rule=MaskRules(
                    rows_above=0, rows_below=0, columns_left=0, columns_right=1
                ),
            ),
            AbstractMask(
                match_symbol="X",
                rule=MaskRules(
                    rows_above=0, rows_below=0, columns_left=1, columns_right=0
                ),
            ),
            AbstractMask(
                match_symbol="O",
                rule=MaskRules(
                    rows_above=0, rows_below=0, columns_left=1, columns_right=0
                ),
            ),
            AbstractMask(
                match_symbol=None,
                rule=MaskRules(
                    rows_above=0, rows_below=0, columns_left=1, columns_right=0
                ),
            ),
            AbstractMask(
                match_symbol="X",
                rule=MaskRules(
                    rows_above=0, rows_below=1, columns_left=0, columns_right=0
                ),
            ),
            AbstractMask(
                match_symbol="O",
                rule=MaskRules(
                    rows_above=0, rows_below=1, columns_left=0, columns_right=0
                ),
            ),
            AbstractMask(
                match_symbol=None,
                rule=MaskRules(
                    rows_above=0, rows_below=1, columns_left=0, columns_right=0
                ),
            ),
            AbstractMask(
                match_symbol="X",
                rule=MaskRules(
                    rows_above=1, rows_below=0, columns_left=0, columns_right=0
                ),
            ),
            AbstractMask(
                match_symbol="O",
                rule=MaskRules(
                    rows_above=1, rows_below=0, columns_left=0, columns_right=0
                ),
            ),
            AbstractMask(
                match_symbol=None,
                rule=MaskRules(
                    rows_above=1, rows_below=0, columns_left=0, columns_right=0
                ),
            ),
            AbstractMask(
                match_symbol="X",
                rule=MaskRules(
                    rows_above=0, rows_below=1, columns_left=0, columns_right=1
                ),
            ),
            AbstractMask(
                match_symbol="O",
                rule=MaskRules(
                    rows_above=0, rows_below=1, columns_left=0, columns_right=1
                ),
            ),
            AbstractMask(
                match_symbol=None,
                rule=MaskRules(
                    rows_above=0, rows_below=1, columns_left=0, columns_right=1
                ),
            ),
            AbstractMask(
                match_symbol="X",
                rule=MaskRules(
                    rows_above=0, rows_below=1, columns_left=1, columns_right=0
                ),
            ),
            AbstractMask(
                match_symbol="O",
                rule=MaskRules(
                    rows_above=0, rows_below=1, columns_left=1, columns_right=0
                ),
            ),
            AbstractMask(
                match_symbol=None,
                rule=MaskRules(
                    rows_above=0, rows_below=1, columns_left=1, columns_right=0
                ),
            ),
            AbstractMask(
                match_symbol="X",
                rule=MaskRules(
                    rows_above=1, rows_below=0, columns_left=0, columns_right=1
                ),
            ),
            AbstractMask(
                match_symbol="O",
                rule=MaskRules(
                    rows_above=1, rows_below=0, columns_left=0, columns_right=1
                ),
            ),
            AbstractMask(
                match_symbol=None,
                rule=MaskRules(
                    rows_above=1, rows_below=0, columns_left=0, columns_right=1
                ),
            ),
            AbstractMask(
                match_symbol="X",
                rule=MaskRules(
                    rows_above=1, rows_below=0, columns_left=1, columns_right=0
                ),
            ),
            AbstractMask(
                match_symbol="O",
                rule=MaskRules(
                    rows_above=1, rows_below=0, columns_left=1, columns_right=0
                ),
            ),
            AbstractMask(
                match_symbol=None,
                rule=MaskRules(
                    rows_above=1, rows_below=0, columns_left=1, columns_right=0
                ),
            ),
        ]

    def test_generates_off_shape_1X2(self):
        masks = generate_pool_masks(1, 2)
        assert len(masks) == 9
        assert masks == [
            AbstractMask(
                match_symbol="X",
                rule=MaskRules(
                    rows_above=0, rows_below=0, columns_left=0, columns_right=0
                ),
            ),
            AbstractMask(
                match_symbol="O",
                rule=MaskRules(
                    rows_above=0, rows_below=0, columns_left=0, columns_right=0
                ),
            ),
            AbstractMask(
                match_symbol=None,
                rule=MaskRules(
                    rows_above=0, rows_below=0, columns_left=0, columns_right=0
                ),
            ),
            AbstractMask(
                match_symbol="X",
                rule=MaskRules(
                    rows_above=0, rows_below=0, columns_left=0, columns_right=1
                ),
            ),
            AbstractMask(
                match_symbol="O",
                rule=MaskRules(
                    rows_above=0, rows_below=0, columns_left=0, columns_right=1
                ),
            ),
            AbstractMask(
                match_symbol=None,
                rule=MaskRules(
                    rows_above=0, rows_below=0, columns_left=0, columns_right=1
                ),
            ),
            AbstractMask(
                match_symbol="X",
                rule=MaskRules(
                    rows_above=0, rows_below=0, columns_left=1, columns_right=0
                ),
            ),
            AbstractMask(
                match_symbol="O",
                rule=MaskRules(
                    rows_above=0, rows_below=0, columns_left=1, columns_right=0
                ),
            ),
            AbstractMask(
                match_symbol=None,
                rule=MaskRules(
                    rows_above=0, rows_below=0, columns_left=1, columns_right=0
                ),
            ),
        ]

    def test_generates_between_sizes(self):
        masks = generate_pool_masks(
            rows=2, columns=2, skip_rows_under=2, skip_columns_under=2
        )
        assert len(masks) == 12
        assert masks == [
            AbstractMask(
                match_symbol="X",
                rule=MaskRules(
                    rows_above=0, rows_below=1, columns_left=0, columns_right=1
                ),
            ),
            AbstractMask(
                match_symbol="O",
                rule=MaskRules(
                    rows_above=0, rows_below=1, columns_left=0, columns_right=1
                ),
            ),
            AbstractMask(
                match_symbol=None,
                rule=MaskRules(
                    rows_above=0, rows_below=1, columns_left=0, columns_right=1
                ),
            ),
            AbstractMask(
                match_symbol="X",
                rule=MaskRules(
                    rows_above=0, rows_below=1, columns_left=1, columns_right=0
                ),
            ),
            AbstractMask(
                match_symbol="O",
                rule=MaskRules(
                    rows_above=0, rows_below=1, columns_left=1, columns_right=0
                ),
            ),
            AbstractMask(
                match_symbol=None,
                rule=MaskRules(
                    rows_above=0, rows_below=1, columns_left=1, columns_right=0
                ),
            ),
            AbstractMask(
                match_symbol="X",
                rule=MaskRules(
                    rows_above=1, rows_below=0, columns_left=0, columns_right=1
                ),
            ),
            AbstractMask(
                match_symbol="O",
                rule=MaskRules(
                    rows_above=1, rows_below=0, columns_left=0, columns_right=1
                ),
            ),
            AbstractMask(
                match_symbol=None,
                rule=MaskRules(
                    rows_above=1, rows_below=0, columns_left=0, columns_right=1
                ),
            ),
            AbstractMask(
                match_symbol="X",
                rule=MaskRules(
                    rows_above=1, rows_below=0, columns_left=1, columns_right=0
                ),
            ),
            AbstractMask(
                match_symbol="O",
                rule=MaskRules(
                    rows_above=1, rows_below=0, columns_left=1, columns_right=0
                ),
            ),
            AbstractMask(
                match_symbol=None,
                rule=MaskRules(
                    rows_above=1, rows_below=0, columns_left=1, columns_right=0
                ),
            ),
        ]


class TestMaskRules:
    @pytest.mark.parametrize(
        "coord, expected",
        [
            ((1, 1), [[X, O, E], [O, E, X], [E, X, O]]),
            ((0, 1), None),
            ((2, 2), None),
            ((1, 0), None),
            ((1, 2), None),
        ],
    )
    def test_get_pattern(
        self, coord: tuple[int, int], expected: list[list[str]] | None
    ) -> None:
        rule = MaskRules(1, 1, 1, 1)
        grid = [
            [X, O, E],
            [O, E, X],
            [E, X, O],
        ]
        assert rule.get_pattern(coord, grid) == expected


class TestAbstractMask:
    def test_remove_non_matching(self) -> None:
        mask = AbstractMask(match_symbol=X, rule=MaskRules(0, 0, 0, 0))
        grid = [[X, O], [E, X]]
        assert mask.remove_non_matching(grid) == [[X, E], [E, X]]

    def test_create_mask_key(self) -> None:
        mask = AbstractMask(match_symbol=None, rule=MaskRules(0, 0, 0, 0))
        value = [[X, E], [O, E]]
        assert mask.create_mask_key(value, X) == MaskKey(
            mask_type=mask, pattern="X_\nO_", symbol=X
        )

    @pytest.mark.parametrize(
        "coord, expected",
        [
            (
                (0, 0),
                MaskKey(
                    mask_type=AbstractMask(match_symbol=X, rule=MaskRules(0, 0, 0, 0)),
                    pattern="X",
                    symbol=X,
                ),
            ),
            ((2, 0), None),
        ],
    )
    def test_get_mask(self, coord: tuple[int, int], expected: MaskKey | None) -> None:
        base_mask = AbstractMask(match_symbol=X, rule=MaskRules(0, 0, 0, 0))
        grid = [[X, O], [E, X]]
        assert base_mask.get_mask(coord, grid, current=X) == expected
