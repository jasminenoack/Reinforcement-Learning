import pytest
from tic_tac_logic.agents.masks import (
    AbstractMaskFactory,
    MaskRules,
    MaskKey,
    CompleteMask,
)
from tic_tac_logic.constants import X, O, E


class TestRemoveNonMatching:
    def test_remove_non_matching(self) -> None:
        mask = AbstractMaskFactory(match_symbol=X, rule=MaskRules(0, 0, 0, 0))
        grid = [[X, O], [E, X]]
        assert mask.remove_non_matching(grid) == [[X, E], [E, X]]


class TestCreateMaskKey:
    def test_create_mask_key(self) -> None:
        mask = AbstractMaskFactory(match_symbol=None, rule=MaskRules(0, 0, 0, 0))
        value = [[X, E], [O, E]]
        assert mask.create_mask_key(value, X) == MaskKey(
            mask_type=mask, pattern="X_\nO_", symbol=X
        )


class TestGetMask:
    @pytest.mark.parametrize(
        "coord, expected",
        [
            (
                (0, 0),
                MaskKey(
                    mask_type=AbstractMaskFactory(
                        match_symbol=X, rule=MaskRules(0, 0, 0, 0)
                    ),
                    pattern="X",
                    symbol=X,
                ),
            ),
            ((2, 0), None),
        ],
    )
    def test_get_mask(self, coord: tuple[int, int], expected: MaskKey | None) -> None:
        base_mask = AbstractMaskFactory(match_symbol=X, rule=MaskRules(0, 0, 0, 0))
        grid = [[X, O], [E, X]]
        assert base_mask.get_mask(coord, grid, current=X) == expected


class TestGenerateMasks:
    def test_generate_masks_horizontal(self) -> None:
        rules = MaskRules(rows_above=0, rows_below=0, columns_left=1, columns_right=1)
        base_mask = AbstractMaskFactory(
            match_symbol=None,
            rule=rules,
        )
        masks = base_mask.generate_masks()
        assert masks == [
            CompleteMask(
                rules=rules,
                pattern="___",
                symbol_to_place=X,
                match_symbol=None,
            ),
            CompleteMask(
                rules=rules,
                pattern="__X",
                symbol_to_place=X,
                match_symbol=None,
            ),
            CompleteMask(
                rules=rules,
                pattern="__O",
                symbol_to_place=X,
                match_symbol=None,
            ),
            CompleteMask(
                rules=rules,
                pattern="X__",
                symbol_to_place=X,
                match_symbol=None,
            ),
            CompleteMask(
                rules=rules,
                pattern="X_X",
                symbol_to_place=X,
                match_symbol=None,
            ),
            CompleteMask(
                rules=rules,
                pattern="X_O",
                symbol_to_place=X,
                match_symbol=None,
            ),
            CompleteMask(
                rules=rules,
                pattern="O__",
                symbol_to_place=X,
                match_symbol=None,
            ),
            CompleteMask(
                rules=rules,
                pattern="O_X",
                symbol_to_place=X,
                match_symbol=None,
            ),
            CompleteMask(
                rules=rules,
                pattern="O_O",
                symbol_to_place=X,
                match_symbol=None,
            ),
            CompleteMask(
                rules=rules,
                pattern="___",
                symbol_to_place=O,
                match_symbol=None,
            ),
            CompleteMask(
                rules=rules,
                pattern="__X",
                symbol_to_place=O,
                match_symbol=None,
            ),
            CompleteMask(
                rules=rules,
                pattern="__O",
                symbol_to_place=O,
                match_symbol=None,
            ),
            CompleteMask(
                rules=rules,
                pattern="X__",
                symbol_to_place=O,
                match_symbol=None,
            ),
            CompleteMask(
                rules=rules,
                pattern="X_X",
                symbol_to_place=O,
                match_symbol=None,
            ),
            CompleteMask(
                rules=rules,
                pattern="X_O",
                symbol_to_place=O,
                match_symbol=None,
            ),
            CompleteMask(
                rules=rules,
                pattern="O__",
                symbol_to_place=O,
                match_symbol=None,
            ),
            CompleteMask(
                rules=rules,
                pattern="O_X",
                symbol_to_place=O,
                match_symbol=None,
            ),
            CompleteMask(
                rules=rules,
                pattern="O_O",
                symbol_to_place=O,
                match_symbol=None,
            ),
        ]
