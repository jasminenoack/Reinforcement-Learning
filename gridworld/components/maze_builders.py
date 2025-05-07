"""
Maze Generation Algorithms (All Ensure Solvability)

Recursive Backtracking (a depth-first search maze)
Very common. Creates long corridors and few branches.
Easy to implement with a stack and visited set.
Carves out a path recursively until stuck, then backtracks.

Prim’s Algorithm (randomized)
Tends to create lots of small paths and loops.
Starts with a grid of walls, then carves paths outward like MST.

Kruskal’s Algorithm (minimum spanning tree)
Ensures no loops.
Treats cells as nodes and randomly connects them without forming cycles.

Aldous-Broder
Random walk that guarantees uniform randomness, but is slow.
No bias in structure; generates very twisty mazes.

Wilson’s Algorithm
Like Aldous-Broder, but faster and still unbiased.
Uses loop-erased random walks to build the maze.
"""

from abc import ABC, abstractmethod
import random


from gridworld.utils import GOAL

START = "S"
GOAL = "G"
OBSTACLE = "#"
PATH = "."
UNKNOWN = "_"


class MazeGenerator(ABC):
    def __init__(
        self,
        *,
        rows: int,
        cols: int,
        start: tuple[int, int] = (0, 0),
        end: tuple[int, int] | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.start = start
        self.end = end or (rows - 1, cols - 1)
        self.rng = rng or random.Random()
        self.grid = [[UNKNOWN] * self.cols for _ in range(self.rows)]

    @abstractmethod
    def run(self) -> list[tuple[int, int]]: ...

    def set_cell(self, coor: tuple[int, int], marker: str) -> None:
        # set the cell to the marker
        if self.grid[coor[0]][coor[1]] == UNKNOWN:
            self.grid[coor[0]][coor[1]] = marker
        else:
            raise ValueError(
                f"Cell {coor} already set to {self.grid[coor[0]][coor[1]]}"
            )

    def get_neighbor_coordinates(self, coor: tuple[int, int]) -> list[tuple[int, int]]:
        neighbors = []
        for dx, dy in [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]:
            x, y = coor[0] + dx, coor[1] + dy
            if 0 <= x < self.rows and 0 <= y < self.cols:
                neighbors.append((x, y))
        return neighbors

    def get_empty_coordinates(self) -> list[tuple[int, int]]:
        empty_cells = []
        for i in range(self.rows):
            for j in range(self.cols):
                if self.grid[i][j] == UNKNOWN:
                    empty_cells.append((i, j))
        return empty_cells

    def get_empty_neighbors(self, coor: tuple[int, int]) -> list[tuple[int, int]]:
        neighbors = set(self.get_neighbor_coordinates(coor))
        empty_cells = set(self.get_empty_coordinates())
        return list(neighbors.intersection(empty_cells))


class SparseObstacleMazeGenerator(MazeGenerator):

    def block(self, coor: tuple, marker: str) -> None:
        self.set_cell(coor, marker)
        neighbors = self.get_empty_neighbors(coor)
        for neighbor in neighbors:
            self.set_cell(neighbor, PATH)

    def run(self) -> dict[str, list[tuple[int, int]]]:
        self.block(self.start, START)
        self.block(self.end, GOAL)
        while empty_cells := self.get_empty_coordinates():
            cell = self.rng.choice(empty_cells)
            self.block(cell, OBSTACLE)
        return {
            "obstacles": [
                (i, j)
                for i in range(self.rows)
                for j in range(self.cols)
                if self.grid[i][j] == OBSTACLE
            ]
        }

    def render(self) -> None:
        for row in self.grid:
            print(" ".join(row))
        print()


class RecursiveBacktracking(MazeGenerator):
    def run(self):

        return []


if __name__ == "__main__":
    maze = SparseObstacleMazeGenerator(
        rows=5,
        cols=5,
        start=(0, 0),
        end=(4, 4),
    )
    maze.run()

    print("sparse maze")
    maze.render()
