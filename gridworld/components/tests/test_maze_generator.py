from gridworld.components.maze_builders import (
    Entry,
    MazeGenerator,
)


PATH = Entry(
    visited=True,
    path=True,
)

UNKNOWN = Entry()

OBSTACLE = Entry(
    visited=True,
    obstacle=True,
)

GOAL = Entry(
    visited=True,
    goal=True,
)

START = Entry(
    visited=True,
    start=True,
)


class Fake(MazeGenerator):
    def run(self) -> list[tuple[int, int]]:
        return [(0, 0), (1, 1), (2, 2)]


class TestGetNeighborCoordinates:
    def test_handles_top_left_corner(self):
        maze = Fake(rows=5, cols=5)
        neighbors = maze.get_neighbor_coordinates((0, 0))
        assert neighbors == [(0, 1), (1, 0), (1, 1)]

    def test_handles_top_right_corner(self):
        maze = Fake(rows=5, cols=5)
        neighbors = maze.get_neighbor_coordinates((0, 4))
        assert neighbors == [(0, 3), (1, 3), (1, 4)]

    def test_handles_bottom_left_corner(self):
        maze = Fake(rows=5, cols=5)
        neighbors = maze.get_neighbor_coordinates((4, 0))
        assert neighbors == [(3, 0), (3, 1), (4, 1)]

    def test_handles_bottom_right_corner(self):
        maze = Fake(rows=5, cols=5)
        neighbors = maze.get_neighbor_coordinates((4, 4))
        assert neighbors == [(3, 3), (3, 4), (4, 3)]

    def test_handles_top_edge(self):
        maze = Fake(rows=5, cols=5)
        neighbors = maze.get_neighbor_coordinates((0, 2))
        assert neighbors == [(0, 1), (0, 3), (1, 1), (1, 2), (1, 3)]

    def test_handles_bottom_edge(self):
        maze = Fake(rows=5, cols=5)
        neighbors = maze.get_neighbor_coordinates((4, 2))
        assert neighbors == [(3, 1), (3, 2), (3, 3), (4, 1), (4, 3)]

    def test_handles_left_edge(self):
        maze = Fake(rows=5, cols=5)
        neighbors = maze.get_neighbor_coordinates((2, 0))
        assert neighbors == [(1, 0), (1, 1), (2, 1), (3, 0), (3, 1)]

    def test_handles_right_edge(self):
        maze = Fake(rows=5, cols=5)
        neighbors = maze.get_neighbor_coordinates((2, 4))
        assert neighbors == [(1, 3), (1, 4), (2, 3), (3, 3), (3, 4)]

    def test_handles_middle(self):
        maze = Fake(rows=5, cols=5)
        neighbors = maze.get_neighbor_coordinates((2, 2))
        assert neighbors == [
            (1, 1),
            (1, 2),
            (1, 3),
            (2, 1),
            (2, 3),
            (3, 1),
            (3, 2),
            (3, 3),
        ]


class TestGetEmptyCoordinates:
    def test_get_empty_coordinates(self):
        maze = Fake(rows=5, cols=5)
        maze.grid = [
            [START, PATH, UNKNOWN, UNKNOWN, OBSTACLE],
            [OBSTACLE, UNKNOWN, UNKNOWN, OBSTACLE, UNKNOWN],
            [UNKNOWN, UNKNOWN, OBSTACLE, UNKNOWN, UNKNOWN],
            [OBSTACLE, UNKNOWN, UNKNOWN, OBSTACLE, UNKNOWN],
            [UNKNOWN, OBSTACLE, UNKNOWN, UNKNOWN, GOAL],
        ]
        empty_coords = maze.get_unvisited_coordinates()
        assert empty_coords == [
            (0, 2),
            (0, 3),
            (1, 1),
            (1, 2),
            (1, 4),
            (2, 0),
            (2, 1),
            (2, 3),
            (2, 4),
            (3, 1),
            (3, 2),
            (3, 4),
            (4, 0),
            (4, 2),
            (4, 3),
        ]


class TestGetEmptyNeighbors:
    def test_get_empty_neighbors(self):
        maze = Fake(rows=5, cols=5)
        maze.grid = [
            [START, PATH, UNKNOWN, UNKNOWN, OBSTACLE],
            [OBSTACLE, PATH, UNKNOWN, OBSTACLE, UNKNOWN],
            [UNKNOWN, UNKNOWN, OBSTACLE, UNKNOWN, UNKNOWN],
            [OBSTACLE, UNKNOWN, UNKNOWN, OBSTACLE, UNKNOWN],
            [UNKNOWN, OBSTACLE, UNKNOWN, UNKNOWN, GOAL],
        ]
        empty_neighbors = maze.get_unvisited_neighbors((2, 2))
        assert empty_neighbors == [(1, 2), (2, 1), (3, 1), (2, 3), (3, 2)]
