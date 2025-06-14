from tic_tac_logic.agents.masks import (
    generate_pool_masks,
    MaskRules,
    CompleteMask,
)


class TestGeneratePoolMasks:
    def test_generates_1x1_mask(self):
        masks = list(generate_pool_masks(1, 1))
        assert masks == [
            CompleteMask(
                match_symbol="X",
                rules=MaskRules(
                    rows_above=0, rows_below=0, columns_left=0, columns_right=0
                ),
                pattern="_",
                symbol_to_place="X",
            ),
            CompleteMask(
                match_symbol="X",
                rules=MaskRules(
                    rows_above=0, rows_below=0, columns_left=0, columns_right=0
                ),
                pattern="_",
                symbol_to_place="O",
            ),
            CompleteMask(
                match_symbol="O",
                rules=MaskRules(
                    rows_above=0, rows_below=0, columns_left=0, columns_right=0
                ),
                pattern="_",
                symbol_to_place="X",
            ),
            CompleteMask(
                match_symbol="O",
                rules=MaskRules(
                    rows_above=0, rows_below=0, columns_left=0, columns_right=0
                ),
                pattern="_",
                symbol_to_place="O",
            ),
            CompleteMask(
                match_symbol=None,
                rules=MaskRules(
                    rows_above=0, rows_below=0, columns_left=0, columns_right=0
                ),
                pattern="_",
                symbol_to_place="X",
            ),
            CompleteMask(
                match_symbol=None,
                rules=MaskRules(
                    rows_above=0, rows_below=0, columns_left=0, columns_right=0
                ),
                pattern="_",
                symbol_to_place="O",
            ),
        ]

    def test_generates_2x2_masks(self):
        masks = list(generate_pool_masks(2, 2))
        assert len(masks) == 406
        assert masks[:14] == [
            CompleteMask(
                match_symbol="X",
                rules=MaskRules(
                    rows_above=0, rows_below=0, columns_left=0, columns_right=0
                ),
                pattern="_",
                symbol_to_place="X",
            ),
            CompleteMask(
                match_symbol="X",
                rules=MaskRules(
                    rows_above=0, rows_below=0, columns_left=0, columns_right=0
                ),
                pattern="_",
                symbol_to_place="O",
            ),
            CompleteMask(
                match_symbol="O",
                rules=MaskRules(
                    rows_above=0, rows_below=0, columns_left=0, columns_right=0
                ),
                pattern="_",
                symbol_to_place="X",
            ),
            CompleteMask(
                match_symbol="O",
                rules=MaskRules(
                    rows_above=0, rows_below=0, columns_left=0, columns_right=0
                ),
                pattern="_",
                symbol_to_place="O",
            ),
            CompleteMask(
                match_symbol=None,
                rules=MaskRules(
                    rows_above=0, rows_below=0, columns_left=0, columns_right=0
                ),
                pattern="_",
                symbol_to_place="X",
            ),
            CompleteMask(
                match_symbol=None,
                rules=MaskRules(
                    rows_above=0, rows_below=0, columns_left=0, columns_right=0
                ),
                pattern="_",
                symbol_to_place="O",
            ),
            CompleteMask(
                match_symbol="X",
                rules=MaskRules(
                    rows_above=0, rows_below=0, columns_left=0, columns_right=1
                ),
                pattern="__",
                symbol_to_place="X",
            ),
            CompleteMask(
                match_symbol="X",
                rules=MaskRules(
                    rows_above=0, rows_below=0, columns_left=0, columns_right=1
                ),
                pattern="_X",
                symbol_to_place="X",
            ),
            CompleteMask(
                match_symbol="X",
                rules=MaskRules(
                    rows_above=0, rows_below=0, columns_left=0, columns_right=1
                ),
                pattern="__",
                symbol_to_place="O",
            ),
            CompleteMask(
                match_symbol="X",
                rules=MaskRules(
                    rows_above=0, rows_below=0, columns_left=0, columns_right=1
                ),
                pattern="_X",
                symbol_to_place="O",
            ),
            CompleteMask(
                match_symbol="O",
                rules=MaskRules(
                    rows_above=0, rows_below=0, columns_left=0, columns_right=1
                ),
                pattern="__",
                symbol_to_place="X",
            ),
            CompleteMask(
                match_symbol="O",
                rules=MaskRules(
                    rows_above=0, rows_below=0, columns_left=0, columns_right=1
                ),
                pattern="_O",
                symbol_to_place="X",
            ),
            CompleteMask(
                match_symbol="O",
                rules=MaskRules(
                    rows_above=0, rows_below=0, columns_left=0, columns_right=1
                ),
                pattern="__",
                symbol_to_place="O",
            ),
            CompleteMask(
                match_symbol="O",
                rules=MaskRules(
                    rows_above=0, rows_below=0, columns_left=0, columns_right=1
                ),
                pattern="_O",
                symbol_to_place="O",
            ),
        ]

    def test_generates_between_sizes(self):
        masks = list(
            generate_pool_masks(
                rows=2, columns=2, skip_rows_under=2, skip_columns_under=2
            )
        )
        assert len(masks) == 344
