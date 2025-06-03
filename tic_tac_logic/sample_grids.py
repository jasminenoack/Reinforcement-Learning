from tic_tac_logic.constants import X, O, E


def get_one_off_grid():
    return [
        [X, O, X, O],
        [O, E, O, X],
        [O, X, X, O],
        [X, O, O, X],
    ]
