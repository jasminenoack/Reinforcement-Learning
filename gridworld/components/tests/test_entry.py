from gridworld.components.maze_builders import (
    START,
    GOAL,
    OBSTACLE,
    PATH,
    UP,
    DOWN,
    LEFT,
    RIGHT,
    Entry,
)


class TestUnknown:
    def test_unknown(self):
        entry = Entry()
        assert entry.unknown
        assert not entry.visited
        assert not entry.path
        assert not entry.obstacle
        assert not entry.start
        assert not entry.goal

    def test_visited(self):
        entry = Entry(visited=True)
        assert entry.visited
        assert not entry.unknown
        assert not entry.path
        assert not entry.obstacle
        assert not entry.start
        assert not entry.goal

    def test_path(self):
        entry = Entry(path=True)
        assert entry.path
        assert not entry.unknown
        assert not entry.visited
        assert not entry.obstacle
        assert not entry.start
        assert not entry.goal

    def test_obstacle(self):
        entry = Entry(obstacle=True)
        assert entry.obstacle
        assert not entry.unknown
        assert not entry.visited
        assert not entry.path
        assert not entry.start
        assert not entry.goal

    def test_start(self):
        entry = Entry(start=True)
        assert entry.start
        assert not entry.unknown
        assert not entry.visited
        assert not entry.path
        assert not entry.obstacle
        assert not entry.goal

    def test_goal(self):
        entry = Entry(goal=True)
        assert entry.goal
        assert not entry.unknown
        assert not entry.visited
        assert not entry.path
        assert not entry.obstacle
        assert not entry.start


class TestMark:
    def test_start(self):
        entry = Entry()
        entry.mark(START)
        assert entry.start
        assert not entry.unknown
        assert entry.visited
        assert not entry.path
        assert not entry.obstacle
        assert not entry.goal

    def test_goal(self):
        entry = Entry()
        entry.mark(GOAL)
        assert entry.goal
        assert not entry.unknown
        assert entry.visited
        assert not entry.path
        assert not entry.obstacle
        assert not entry.start

    def test_obstacle(self):
        entry = Entry()
        entry.mark(OBSTACLE)
        assert entry.obstacle
        assert not entry.unknown
        assert entry.visited
        assert not entry.path
        assert not entry.start
        assert not entry.goal

    def test_path(self):
        entry = Entry()
        entry.mark(PATH)
        assert entry.path
        assert not entry.unknown
        assert entry.visited
        assert not entry.obstacle
        assert not entry.start
        assert not entry.goal

    def test_up(self):
        entry = Entry()
        entry.mark(UP)
        assert entry.direction == UP
        assert not entry.unknown
        assert entry.visited
        assert entry.path
        assert not entry.obstacle
        assert not entry.start
        assert not entry.goal

    def test_down(self):
        entry = Entry()
        entry.mark(DOWN)
        assert entry.direction == DOWN
        assert not entry.unknown
        assert entry.visited
        assert entry.path
        assert not entry.obstacle
        assert not entry.start
        assert not entry.goal

    def test_left(self):
        entry = Entry()
        entry.mark(LEFT)
        assert entry.direction == LEFT
        assert not entry.unknown
        assert entry.visited
        assert entry.path
        assert not entry.obstacle
        assert not entry.start
        assert not entry.goal

    def test_right(self):
        entry = Entry()
        entry.mark(RIGHT)
        assert entry.direction == RIGHT
        assert not entry.unknown
        assert entry.visited
        assert entry.path
        assert not entry.obstacle
        assert not entry.start
        assert not entry.goal


class TestRender:
    def test_start(self):
        entry = Entry(start=True)
        assert entry.render() == "S"

    def test_goal(self):
        entry = Entry(goal=True)
        assert entry.render() == "G"

    def test_obstacle(self):
        entry = Entry(obstacle=True)
        assert entry.render() == "#"

    def test_visited(self):
        entry = Entry(visited=True)
        assert entry.render() == " "

    def test_direction(self):
        entry = Entry(direction=UP)
        assert entry.render() == UP
