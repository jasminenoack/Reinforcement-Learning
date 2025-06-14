import pytest
from tic_tac_logic.agents.masks import AbstractMaskFactory, MaskRules, MaskKey
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
