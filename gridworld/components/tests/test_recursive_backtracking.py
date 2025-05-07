import random
from gridworld.components.maze_builders import (
    RecursiveBacktracking,
    CellSet,
    Entry,
    Walls,
    UP,
    DOWN,
    LEFT,
    RIGHT,
)


class TestKoolaidMan:
    def test_handles_new_on_top(self):
        cell_1 = Entry(walls=Walls(up=True, down=True, left=True, right=True))
        cell_2 = Entry(walls=Walls(up=True, down=True, left=True, right=True))
        cell_set = CellSet(
            cell=cell_1,
            coor=(0, 0),
            last_cell=cell_2,
            last_coor=(1, 0),
        )
        maze = RecursiveBacktracking(
            rows=5,
            cols=5,
            start=(0, 0),
            end=(4, 4),
        )
        maze.koolaid_man(cell_set)
        assert cell_1.walls == Walls(up=True, down=False, left=True, right=True)
        assert cell_2.walls == Walls(up=False, down=True, left=True, right=True)
        assert cell_2.direction == UP

    def test_handles_new_on_bottom(self):
        cell_1 = Entry(walls=Walls(up=True, down=True, left=True, right=True))
        cell_2 = Entry(walls=Walls(up=True, down=True, left=True, right=True))
        cell_set = CellSet(
            cell=cell_1,
            coor=(1, 0),
            last_cell=cell_2,
            last_coor=(0, 0),
        )
        maze = RecursiveBacktracking(
            rows=5,
            cols=5,
            start=(0, 0),
            end=(4, 4),
        )
        maze.koolaid_man(cell_set)
        assert cell_1.walls == Walls(up=False, down=True, left=True, right=True)
        assert cell_2.walls == Walls(up=True, down=False, left=True, right=True)
        assert cell_2.direction == DOWN

    def test_handles_new_on_left(self):
        cell_1 = Entry(walls=Walls(up=True, down=True, left=True, right=True))
        cell_2 = Entry(walls=Walls(up=True, down=True, left=True, right=True))
        cell_set = CellSet(
            cell=cell_1,
            coor=(0, 0),
            last_cell=cell_2,
            last_coor=(0, 1),
        )
        maze = RecursiveBacktracking(
            rows=5,
            cols=5,
            start=(0, 0),
            end=(4, 4),
        )
        maze.koolaid_man(cell_set)
        assert cell_1.walls == Walls(up=True, down=True, left=True, right=False)
        assert cell_2.walls == Walls(up=True, down=True, left=False, right=True)
        assert cell_2.direction == LEFT

    def test_handles_new_on_right(self):
        cell_1 = Entry(walls=Walls(up=True, down=True, left=True, right=True))
        cell_2 = Entry(walls=Walls(up=True, down=True, left=True, right=True))
        cell_set = CellSet(
            cell=cell_1,
            coor=(0, 1),
            last_cell=cell_2,
            last_coor=(0, 0),
        )
        maze = RecursiveBacktracking(
            rows=5,
            cols=5,
            start=(0, 0),
            end=(4, 4),
        )
        maze.koolaid_man(cell_set)
        assert cell_1.walls == Walls(up=True, down=True, left=False, right=True)
        assert cell_2.walls == Walls(up=True, down=True, left=True, right=False)
        assert cell_2.direction == RIGHT


class TestRun:
    def test_generates_maze(self):
        rng = random.Random(42)
        maze = RecursiveBacktracking(
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
                    path=True,
                    walls=Walls(
                        up=True,
                        down=False,
                        left=True,
                        right=False,
                    ),
                    direction="↓",
                ),
                Entry(
                    visited=True,
                    start=False,
                    goal=False,
                    obstacle=False,
                    path=True,
                    walls=Walls(
                        up=True,
                        down=False,
                        left=False,
                        right=False,
                    ),
                    direction="→",
                ),
                Entry(
                    visited=True,
                    start=False,
                    goal=False,
                    obstacle=False,
                    path=True,
                    walls=Walls(
                        up=True,
                        down=True,
                        left=False,
                        right=True,
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
                        left=True,
                        right=True,
                    ),
                    direction="↓",
                ),
                Entry(
                    visited=True,
                    start=False,
                    goal=False,
                    obstacle=False,
                    path=True,
                    walls=Walls(
                        up=False,
                        down=True,
                        left=True,
                        right=True,
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
                        up=True,
                        down=False,
                        left=True,
                        right=True,
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
                        down=True,
                        left=True,
                        right=False,
                    ),
                    direction="→",
                ),
                Entry(
                    visited=True,
                    start=False,
                    goal=False,
                    obstacle=False,
                    path=True,
                    walls=Walls(
                        up=True,
                        down=True,
                        left=False,
                        right=False,
                    ),
                    direction="→",
                ),
                Entry(
                    visited=True,
                    start=False,
                    goal=True,
                    obstacle=False,
                    path=True,
                    walls=Walls(
                        up=False,
                        down=True,
                        left=False,
                        right=True,
                    ),
                    direction="↑",
                ),
            ],
        ]
