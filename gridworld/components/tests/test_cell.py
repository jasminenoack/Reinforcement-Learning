from gridworld.components.grid_environment import Cell
from gridworld.components.maze_builders import Walls
from gridworld.utils import DOWN, LEFT, RIGHT, UP


class TestObstacle:
    def test_obstacle(self):
        cell = Cell(_obstacle=True)
        assert cell.obstacle is True

    def test_not_obstacle(self):
        cell = Cell(_obstacle=False)
        assert cell.obstacle is False


class TestSetObstacle:
    def test_set_obstacle(self):
        cell = Cell()
        cell.obstacle = True
        assert cell.obstacle is True

    def test_set_not_obstacle(self):
        cell = Cell(_obstacle=True)
        cell.obstacle = False
        assert cell.obstacle is False


class TestHasDoor:
    def test_has_door_up(self):
        cell = Cell(walls=Walls(up=False, down=True, left=True, right=True))
        assert cell.has_door(UP) is True
        assert cell.has_door(DOWN) is False
        assert cell.has_door(LEFT) is False
        assert cell.has_door(RIGHT) is False

    def test_has_door_down(self):
        cell = Cell(walls=Walls(up=True, down=False, left=True, right=True))
        assert cell.has_door(UP) is False
        assert cell.has_door(DOWN) is True
        assert cell.has_door(LEFT) is False
        assert cell.has_door(RIGHT) is False

    def test_has_door_left(self):
        cell = Cell(walls=Walls(up=True, down=True, left=False, right=True))
        assert cell.has_door(UP) is False
        assert cell.has_door(DOWN) is False
        assert cell.has_door(LEFT) is True
        assert cell.has_door(RIGHT) is False

    def test_has_door_right(self):
        cell = Cell(walls=Walls(up=True, down=True, left=True, right=False))
        assert cell.has_door(UP) is False
        assert cell.has_door(DOWN) is False
        assert cell.has_door(LEFT) is False
        assert cell.has_door(RIGHT) is True

    def test_no_walls(self):
        cell = Cell(walls=None)
        assert cell.has_door(UP) is True
        assert cell.has_door(DOWN) is True
        assert cell.has_door(LEFT) is True
        assert cell.has_door(RIGHT) is True
