from tic_tac_logic.env.grid import Grid
from tic_tac_logic.sample_grids import get_one_off_grid


def print_grid(grid: Grid) -> None:
    for row in grid.grid:
        print(" | ".join(row))
    print()


if __name__ == "__main__":
    grid = Grid(get_one_off_grid())
    print("Initial Grid:")
    print_grid(grid)
