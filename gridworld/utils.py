from typing import NamedTuple, TypedDict

from matplotlib import pyplot as plt
import numpy as np


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
    visit_counts: dict[tuple[int, int], int]


def render_heatmap(
    *,
    visit_counts: dict[tuple[int, int], int],
    rows: int,
    cols: int,
    stat="Visit Count",
    folder: str = "output",
) -> None:
    numpy_map = np.zeros((rows, cols))
    for (x, y), count in visit_counts.items():
        numpy_map[x, y] = count

    name = f"Gridworld {stat} Heatmap"
    plt.imshow(numpy_map, cmap="hot", interpolation="nearest")
    plt.colorbar(label=stat)
    plt.title(name)
    plt.show(block=False)
    plt.pause(2)
    plt.savefig(
        f'{folder}/{name.lower().replace(" ", "_").replace("(", "_").replace(")", "_")}.png'
    )
    plt.close()
