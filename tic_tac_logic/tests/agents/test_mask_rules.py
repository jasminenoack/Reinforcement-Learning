import pytest
from tic_tac_logic.agents.masks import MaskRules
from tic_tac_logic.constants import X, O, E


class TestGetPattern:
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
