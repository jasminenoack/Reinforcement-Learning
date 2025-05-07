from re import L
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
    filename: str | None = None,
    scale_max: int = 20,
) -> None:
    numpy_map = np.zeros((rows, cols))
    for (x, y), count in visit_counts.items():
        numpy_map[x, y] = count

    name = f"Gridworld {stat} Heatmap"
    plt.imshow(numpy_map, cmap="hot", interpolation="nearest", vmax=scale_max)
    plt.colorbar(label=stat)
    plt.title(name)
    plt.show(block=False)
    plt.pause(2)
    filename = (
        filename
        or f'{name.lower().replace(" ", "_").replace("(", "_").replace(")", "_")}'
    )
    plt.savefig(f"{folder}/{filename}.png")
    plt.close()


def render_directional_heatmap_for_q_table(
    *,
    visit_counts: dict[tuple[int, int], int],
    rows: int,
    cols: int,
    q_table: dict[tuple[int, int], dict[str, float]],
    stat="Favorite Direction",
    folder: str = "output",
    filename: str | None = None,
    scale_max: int = 20,
) -> None:
    all_confidences = [
        max(actions.values()) - sorted(actions.values())[-2]
        for actions in q_table.values()
        if len(set(actions.values())) > 1
    ]
    max_conf = max(all_confidences) if all_confidences else 1.0

    numpy_map = np.zeros((rows, cols))
    for (x, y), count in visit_counts.items():
        numpy_map[x, y] = count

    name = f"Gridworld {stat} Heatmap"
    plt.imshow(numpy_map, cmap="hot", interpolation="nearest", vmax=scale_max)
    for (x, y), actions in q_table.items():
        q_values = sorted(list(actions.values()))
        max_action_value = q_values[-1]
        second_highest_action_value = q_values[-2]
        confidence = (max_action_value - second_highest_action_value) / max_conf

        if confidence == 0:
            continue
        scale = max(0.2, min(confidence, 1.0))

        max_actions = [
            action for action, value in actions.items() if value == max_action_value
        ]

        if max_action_value < 0:
            color = "red"
        elif max_action_value > 0:
            color = "green"
        else:
            color = "yellow"

        if len(max_actions) > 1:
            continue
        if UP in max_actions:
            plt.arrow(
                y,
                x,
                0,
                -scale,
                head_width=0.1 * scale,
                head_length=0.2 * scale,
                fc=color,
                ec=color,
            )
        elif DOWN in max_actions:
            plt.arrow(
                y,
                x,
                0,
                scale,
                head_width=0.1 * scale,
                head_length=0.2 * scale,
                fc=color,
                ec=color,
            )
        elif LEFT in max_actions:
            plt.arrow(
                y,
                x,
                -scale,
                0,
                head_width=0.1 * scale,
                head_length=0.2 * scale,
                fc=color,
                ec=color,
            )
        elif RIGHT in max_actions:
            plt.arrow(
                y,
                x,
                scale,
                0,
                head_width=0.1 * scale,
                head_length=0.2 * scale,
                fc=color,
                ec=color,
            )
    plt.colorbar(label=stat)
    plt.title(name)
    plt.show(block=False)
    plt.pause(2)
    filename = (
        filename
        or f'{name.lower().replace(" ", "_").replace("(", "_").replace(")", "_")}'
    )
    plt.savefig(f"{folder}/{filename}.png")
    plt.close()
