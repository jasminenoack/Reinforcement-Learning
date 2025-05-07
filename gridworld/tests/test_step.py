from gridworld.utils import DOWN, LEFT, RIGHT, UP, Step


class TestGetReverseAction:
    def test_get_reverse_action_from_up(self):
        step = Step(
            start=(1, 0),
            action=UP,
            new_state=(0, 0),
            reward=0,
            done=False,
        )
        assert step.get_reverse_action() == Step(
            start=(0, 0),
            action=DOWN,
            new_state=(1, 0),
            reward=0,
            done=False,
        )

    def test_get_reverse_action_from_down(self):
        step = Step(
            start=(0, 0),
            action=DOWN,
            new_state=(1, 0),
            reward=0,
            done=False,
        )
        assert step.get_reverse_action() == Step(
            start=(1, 0),
            action=UP,
            new_state=(0, 0),
            reward=0,
            done=False,
        )

    def test_get_reverse_action_from_left(self):
        step = Step(
            start=(0, 1),
            action=LEFT,
            new_state=(0, 0),
            reward=0,
            done=False,
        )
        assert step.get_reverse_action() == Step(
            start=(0, 0),
            action=RIGHT,
            new_state=(0, 1),
            reward=0,
            done=False,
        )

    def test_get_reverse_action_from_right(self):
        step = Step(
            start=(0, 0),
            action=RIGHT,
            new_state=(0, 1),
            reward=0,
            done=False,
        )
        assert step.get_reverse_action() == Step(
            start=(0, 1),
            action=LEFT,
            new_state=(0, 0),
            reward=0,
            done=False,
        )

    def test_handles_known_reverse_action(self):
        step = Step(
            start=(0, 0),
            action="other",
            new_state=(0, 1),
            reward=0,
            done=False,
        )
        assert step.get_reverse_action() is None

    def test_returns_none_if_no_change_in_state(self):
        step = Step(
            start=(0, 0),
            action="other",
            new_state=(0, 0),
            reward=0,
            done=False,
        )
        assert step.get_reverse_action() is None
