from tic_tac_logic.constants import X, O, E


def get_one_off_grid():
    return [
        [X, O, X, O],
        [O, E, O, X],
        [O, X, X, O],
        [X, O, O, X],
    ]


def get_easy_grid():
    return [
        [E, E, E, E, E, E],
        [X, X, E, E, O, E],
        [E, E, O, E, E, O],
        [E, X, E, E, E, E],
        [E, E, E, E, O, E],
        [X, E, X, E, E, E],
        [X, E, O, E, E, X],
        [E, E, E, E, E, E],
    ]
