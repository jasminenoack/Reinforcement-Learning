from typing import NamedTuple, TypedDict


UP = "up"
DOWN = "down"
LEFT = "left"
RIGHT = "right"

SIMPLE_ACTIONS = [UP, DOWN, LEFT, RIGHT]


class Step(NamedTuple):
    start: tuple[int, int]
    action: str
    reward: float
    new_state: tuple[int, int]
    done: bool


class RunnerReturn(TypedDict):
    total_reward: float
    steps: int
    reached_goal: bool
    trajectory: list[tuple[int, int]]
