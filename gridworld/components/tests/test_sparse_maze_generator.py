import random
from gridworld.components.maze_builders import (
    Entry,
    SparseObstacleMazeGenerator,
    Walls,
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
            rows=3,
            cols=3,
            start=(0, 0),
            rng=rng,
        ).run()
        assert maze == [
            [
                Entry(
                    visited=True,
                    start=True,
                    goal=False,
                    obstacle=False,
                    path=False,
                    walls=Walls(
                        up=False,
                        down=False,
                        left=False,
                        right=False,
                    ),
                    direction=None,
                ),
                Entry(
                    visited=True,
                    start=False,
                    goal=False,
                    obstacle=False,
                    path=True,
                    walls=Walls(
                        up=False,
                        down=False,
                        left=False,
                        right=False,
                    ),
                    direction=None,
                ),
                Entry(
                    visited=True,
                    start=False,
                    goal=False,
                    obstacle=True,
                    path=False,
                    walls=Walls(
                        up=False,
                        down=False,
                        left=False,
                        right=False,
                    ),
                    direction=None,
                ),
            ],
            [
                Entry(
                    visited=True,
                    start=False,
                    goal=False,
                    obstacle=False,
                    path=True,
                    walls=Walls(
                        up=False,
                        down=False,
                        left=False,
                        right=False,
                    ),
                    direction=None,
                ),
                Entry(
                    visited=True,
                    start=False,
                    goal=False,
                    obstacle=False,
                    path=True,
                    walls=Walls(
                        up=False,
                        down=False,
                        left=False,
                        right=False,
                    ),
                    direction=None,
                ),
                Entry(
                    visited=True,
                    start=False,
                    goal=False,
                    obstacle=False,
                    path=True,
                    walls=Walls(
                        up=False,
                        down=False,
                        left=False,
                        right=False,
                    ),
                    direction=None,
                ),
            ],
            [
                Entry(
                    visited=True,
                    start=False,
                    goal=False,
                    obstacle=True,
                    path=False,
                    walls=Walls(
                        up=False,
                        down=False,
                        left=False,
                        right=False,
                    ),
                    direction=None,
                ),
                Entry(
                    visited=True,
                    start=False,
                    goal=False,
                    obstacle=False,
                    path=True,
                    walls=Walls(
                        up=False,
                        down=False,
                        left=False,
                        right=False,
                    ),
                    direction=None,
                ),
                Entry(
                    visited=True,
                    start=False,
                    goal=True,
                    obstacle=False,
                    path=False,
                    walls=Walls(
                        up=False,
                        down=False,
                        left=False,
                        right=False,
                    ),
                    direction=None,
                ),
            ],
        ]
