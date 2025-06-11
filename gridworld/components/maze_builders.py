"""
Maze Generation Algorithms (All Ensure Solvability)

- [X] Recursive Backtracking (a depth-first search maze)
Very common. Creates long corridors and few branches.
Easy to implement with a stack and visited set.
Carves out a path recursively until stuck, then backtracks.

- [ ] Prim’s Algorithm (randomized)
Tends to create lots of small paths and loops.
Starts with a grid of walls, then carves paths outward like MST.

- [ ] Kruskal’s Algorithm (minimum spanning tree)
Ensures no loops.
Treats cells as nodes and randomly connects them without forming cycles.

- [ ] Aldous-Broder
Random walk that guarantees uniform randomness, but is slow.
No bias in structure; generates very twisty mazes.

- [ ] Wilson’s Algorithm
Like Aldous-Broder, but faster and still unbiased.
Uses loop-erased random walks to build the maze.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import random

from numpy import diag


from gridworld.utils import GOAL

START = "S"
GOAL = "G"
OBSTACLE = "#"
PATH = " "
UNKNOWN = "_"

UP_DELTA = (-1, 0)
DOWN_DELTA = (1, 0)
LEFT_DELTA = (0, -1)
RIGHT_DELTA = (0, 1)
UP_LEFT_DELTA_DIAG = (-1, -1)
DOWN_LEFT_DELTA_DIAG = (1, 1)
UP_RIGHT_DELTA_DIAG = (-1, 1)
DOWN_RIGHT_DELTA_DIAG = (1, -1)

UP = "↑"
DOWN = "↓"
LEFT = "←"
RIGHT = "→"
UP_RIGHT = "↗"
UP_LEFT = "↖"
DOWN_RIGHT = "↘"
DOWN_LEFT = "↙"


@dataclass
class Walls:
    up: bool = False
    down: bool = False
    left: bool = False
    right: bool = False


@dataclass
class Entry:
    visited: bool = False
    start: bool = False
    goal: bool = False
    obstacle: bool = False
    path: bool = False
    walls: Walls = field(default_factory=Walls)
    direction: str | None = None

    @property
    def unknown(self) -> bool:
        return not (
            self.visited or self.start or self.goal or self.obstacle or self.path
        )

    def mark(self, marker: str, visited: bool = True) -> None:
        self.visited = visited
        if marker == START:
            self.start = True
        elif marker == GOAL:
            self.goal = True
        elif marker == OBSTACLE:
            self.obstacle = True
        elif marker in [UP, DOWN, LEFT, RIGHT]:
            self.path = True
            self.direction = marker
        elif marker == PATH:
            self.path = True
        else:
            raise ValueError(f"Unknown marker: {marker}")

    def render(self) -> str:
        if self.start:
            return START
        elif self.goal:
            return GOAL
        elif self.obstacle:
            return OBSTACLE
        elif self.direction:
            return self.direction
        elif self.visited or self.path:
            return PATH
        else:
            return UNKNOWN


class MazeGenerator(ABC):
    def __init__(
        self,
        *,
        rows: int,
        cols: int,
        start: tuple[int, int] = (0, 0),
        end: tuple[int, int] | None = None,
        rng: random.Random | None = None,
        walls: bool = False,
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.start = start
        self.end = end or (rows - 1, cols - 1)
        self.rng = rng or random.Random()
        self.grid = []
        for i in range(rows):
            row = []
            for j in range(cols):
                row.append(
                    Entry(
                        walls=Walls(
                            up=walls,
                            down=walls,
                            left=walls,
                            right=walls,
                        )
                    )
                )
            self.grid.append(row)

    @abstractmethod
    def run(self) -> list[list[Entry]]: ...

    def get_cell(self, coor: tuple[int, int]) -> Entry:
        # get the cell at the given coordinates
        if not (0 <= coor[0] < self.rows and 0 <= coor[1] < self.cols):
            raise ValueError(f"Coordinates {coor} are out of bounds.")
        return self.grid[coor[0]][coor[1]]

    def set_cell(
        self, coor: tuple[int, int], marker: str, visited: bool = True
    ) -> None:
        # set the cell to the marker
        cell = self.get_cell(coor)
        if not self.get_cell(coor).visited:
            self.get_cell(coor).visited
            self.get_cell(coor).mark(marker, visited=visited)
        else:
            raise ValueError(f"Cell {coor} already set to {self.get_cell(coor)}")

    def get_neighbor_coordinates(
        self, coor: tuple[int, int], orthogonal: bool = False
    ) -> list[tuple[int, int]]:
        deltas = [
            UP_DELTA,
            DOWN_DELTA,
            LEFT_DELTA,
            RIGHT_DELTA,
        ]
        if not orthogonal:
            deltas.extend(
                [
                    UP_LEFT_DELTA_DIAG,
                    UP_RIGHT_DELTA_DIAG,
                    DOWN_LEFT_DELTA_DIAG,
                    DOWN_RIGHT_DELTA_DIAG,
                ]
            )

        neighbors = []
        for dx, dy in deltas:
            x, y = coor[0] + dx, coor[1] + dy
            if 0 <= x < self.rows and 0 <= y < self.cols:
                neighbors.append((x, y))
        return sorted(neighbors)

    def get_unvisited_coordinates(self) -> list[tuple[int, int]]:
        empty_cells = []
        for i in range(self.rows):
            for j in range(self.cols):
                if not self.grid[i][j].visited:
                    empty_cells.append((i, j))
        return empty_cells

    def get_unvisited_neighbors(self, coor: tuple[int, int]) -> list[tuple[int, int]]:
        neighbors = set(self.get_neighbor_coordinates(coor))
        empty_cells = set(self.get_unvisited_coordinates())
        return list(neighbors.intersection(empty_cells))

    def get_unvisited_orthogonal_neighbors(
        self, coor: tuple[int, int]
    ) -> list[tuple[int, int]]:
        neighbors = set(self.get_neighbor_coordinates(coor, orthogonal=True))
        empty_cells = set(self.get_unvisited_coordinates())
        return list(neighbors.intersection(empty_cells))

    def render(self) -> None:
        for i, row in enumerate(self.grid):
            # Top walls
            top_line = ""
            for j, cell in enumerate(row):
                top_line += "+"
                top_line += "---" if cell.walls.up else "   "
            top_line += "+"
            print(top_line)

            # Side walls + cell contents
            mid_line = ""
            for j, cell in enumerate(row):
                mid_line += "|" if cell.walls.left else " "
                mid_line += f" {cell.render()} "
            # Right wall of last cell
            mid_line += "|" if row[-1].walls.right else " "
            print(mid_line)

        # Bottom wall for last row
        bottom_line = ""
        for cell in self.grid[-1]:
            bottom_line += "+"
            bottom_line += "---" if cell.walls.down else "   "
        bottom_line += "+"
        print(bottom_line)


class SparseObstacleMazeGenerator(MazeGenerator):

    def block(self, coor: tuple, marker: str) -> None:
        self.set_cell(coor, marker)
        neighbors = self.get_unvisited_neighbors(coor)
        for neighbor in neighbors:
            self.set_cell(neighbor, PATH)

    def run(self) -> list[list[Entry]]:
        self.block(self.start, START)
        self.block(self.end, GOAL)
        i = 0
        while empty_cells := self.get_unvisited_coordinates():
            cell = self.rng.choice(empty_cells)
            self.block(cell, OBSTACLE)
            i += 1

        return self.grid


@dataclass
class CellSet:
    cell: Entry
    coor: tuple[int, int]
    last_coor: tuple[int, int] | None = None
    last_cell: Entry | None = None


class RecursiveBacktracking(MazeGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, walls=True, **kwargs)

    def koolaid_man(self, cell_data: CellSet) -> None:
        previous_cell = cell_data.last_cell
        current_cell = cell_data.cell
        previous_coor = cell_data.last_coor
        current_coor = cell_data.coor

        cell_1_direction_delta = (
            current_coor[0] - previous_coor[0],
            current_coor[1] - previous_coor[1],
        )
        # Determine direction from previous cell to current cell using coordinate deltas.
        # The deltas are calculated as (current_row - previous_row, current_col - previous_col)
        # and matched to directional movement:
        #
        #   UP_DELTA    = (-1, 0)  → current is above previous
        #   DOWN_DELTA  = (1, 0)   → current is below previous
        #   LEFT_DELTA  = (0, -1)  → current is left of previous
        #   RIGHT_DELTA = (0, 1)   → current is right of previous
        #
        # Based on the movement direction, we:
        # - Knock down the wall in the *current* cell facing the previous cell
        # - Knock down the corresponding wall in the *previous* cell facing the current cell
        if cell_1_direction_delta == UP_DELTA:
            current_cell.walls.down = False
            previous_cell.walls.up = False
            previous_cell.mark(UP)
        elif cell_1_direction_delta == DOWN_DELTA:
            current_cell.walls.up = False
            previous_cell.walls.down = False
            previous_cell.mark(DOWN)
        elif cell_1_direction_delta == LEFT_DELTA:
            current_cell.walls.right = False
            previous_cell.walls.left = False
            previous_cell.mark(LEFT)
        elif cell_1_direction_delta == RIGHT_DELTA:
            current_cell.walls.left = False
            previous_cell.walls.right = False
            previous_cell.mark(RIGHT)

    def run(self) -> list[list[Entry]]:
        self.set_cell(self.start, START, visited=False)
        self.set_cell(self.end, GOAL, visited=False)
        unvisited_cells = self.get_unvisited_coordinates()
        cell_coor = self.rng.choice(unvisited_cells)
        self.set_cell(cell_coor, PATH)
        stack = [CellSet(self.get_cell(cell_coor), cell_coor)]

        while stack:
            # for i in range(10):
            cell_data = stack.pop()
            cell = cell_data.cell
            cell_location = cell_data.coor
            unvisited_neighbors = self.get_unvisited_orthogonal_neighbors(cell_location)
            self.rng.shuffle(unvisited_neighbors)
            stack.extend(
                [
                    CellSet(
                        cell=self.get_cell(neighbor),
                        coor=neighbor,
                        last_coor=cell_location,
                        last_cell=cell,
                    )
                    for neighbor in unvisited_neighbors
                ]
            )
            for neighbor in unvisited_neighbors:
                self.set_cell(neighbor, PATH)
            if cell_data.last_coor is not None:
                self.koolaid_man(cell_data)

        return self.grid


if __name__ == "__main__":
    sparse_maze = SparseObstacleMazeGenerator(
        rows=10,
        cols=10,
        start=(0, 0),
    )
    sparse_maze.run()
    print("Sparse Maze:")
    sparse_maze.render()

    recursive_maze = RecursiveBacktracking(
        rows=10,
        cols=10,
        start=(0, 0),
    )
    recursive_maze.run()
    print("Recursive Backtracking Maze:")
    recursive_maze.render()
