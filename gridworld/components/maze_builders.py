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
from dataclasses import dataclass
import random
from tracemalloc import start


from gridworld.utils import GOAL

START = "S"
GOAL = "G"
OBSTACLE = "#"
PATH = "."
UNKNOWN = "_"


@dataclass
class Entry:
    visited: bool = False
    start: bool = False
    goal: bool = False
    obstacle: bool = False
    path: bool = False

    def mark(self, marker: str) -> None:
        self.visited = True
        if marker == START:
            self.start = True
        elif marker == GOAL:
            self.goal = True
        elif marker == OBSTACLE:
            self.obstacle = True
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
        elif self.visited:
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
                row.append(Entry())
            self.grid.append(row)

    @abstractmethod
    def run(self) -> list[tuple[int, int]]: ...

    def set_cell(self, coor: tuple[int, int], marker: str) -> None:
        # set the cell to the marker
        cell = self.grid[coor[0]][coor[1]]
        if not self.grid[coor[0]][coor[1]].visited:
            self.grid[coor[0]][coor[1]].visited
            self.grid[coor[0]][coor[1]].mark(marker)
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


class SparseObstacleMazeGenerator(MazeGenerator):

    def block(self, coor: tuple, marker: str) -> None:
        self.set_cell(coor, marker)
        neighbors = self.get_unvisited_neighbors(coor)
        for neighbor in neighbors:
            self.set_cell(neighbor, PATH)

    def run(self) -> dict[str, list[tuple[int, int]]]:
        self.block(self.start, START)
        self.block(self.end, GOAL)
        i = 0
        while empty_cells := self.get_unvisited_coordinates():
            cell = self.rng.choice(empty_cells)
            self.block(cell, OBSTACLE)
            i += 1

        return {
            "obstacles": [
                (i, j)
                for i in range(self.rows)
                for j in range(self.cols)
                if self.grid[i][j].obstacle
            ]
        }

    def render(self) -> None:
        for row in self.grid:
            print(" ".join(item.render() for item in row))
        print()


class RecursiveBacktracking(MazeGenerator):
    def run(self):

        return []


if __name__ == "__main__":
    maze = SparseObstacleMazeGenerator(
        rows=10,
        cols=10,
        start=(0, 0),
    )
    maze.run()

    maze.render()
