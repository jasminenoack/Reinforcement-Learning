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
