import random
from gridworld.components.maze_builders import (
    Entry,
    SparseObstacleMazeGenerator,
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


class TestBlock:
    def test_block(self):
        rng = random.Random(42)
        maze = SparseObstacleMazeGenerator(
            rows=5,
            cols=5,
            start=(0, 0),
            end=(4, 4),
            rng=rng,
        )
        maze.block((1, 1), "#")
        assert maze.grid == [
            [PATH, PATH, PATH, UNKNOWN, UNKNOWN],
            [PATH, OBSTACLE, PATH, UNKNOWN, UNKNOWN],
            [PATH, PATH, PATH, UNKNOWN, UNKNOWN],
            [UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN],
            [UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN],
        ]


class TestSparseMazeGenerator:
    def test_sparse_maze_generator(self):
        rng = random.Random(42)
        maze = SparseObstacleMazeGenerator(
            rows=5,
            cols=5,
            start=(0, 0),
            end=(4, 4),
            rng=rng,
        ).run()
        assert maze == {
            "obstacles": [
                (0, 4),
                (1, 2),
                (2, 0),
                (2, 4),
                (3, 2),
                (4, 0),
            ]
        }
