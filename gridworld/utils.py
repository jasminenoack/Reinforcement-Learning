from typing import NamedTuple, TypedDict

from matplotlib import pyplot as plt
from matplotlib.table import Cell
import numpy as np


UP = "up"
DOWN = "down"
LEFT = "left"
RIGHT = "right"

SIMPLE_ACTIONS = [UP, DOWN, LEFT, RIGHT]

OFF_BOARD = "off_board"
OBSTACLE = "obstacle"
GOAL = "goal"
MOVEMENT = "movement"
INTERIOR_WALL = "interior_wall"

REVERSED_ACTIONS = {
    UP: DOWN,
    DOWN: UP,
    LEFT: RIGHT,
    RIGHT: LEFT,
}


class Step(NamedTuple):
    start: tuple[int, int]
    action: str
    reward: float
    new_state: tuple[int, int]
    done: bool

    def get_reverse_action(self) -> "Step":
        if self.start == self.new_state:
            return None
        reverse_actions = {
            UP: DOWN,
            DOWN: UP,
            LEFT: RIGHT,
            RIGHT: LEFT,
        }
        if self.action in reverse_actions:
            return Step(
                start=self.new_state,
                action=reverse_actions[self.action],
                reward=self.reward,
                new_state=self.start,
                done=self.done,
            )
        return None


class RunnerReturn(TypedDict):
    total_reward: float
    steps: int
    reached_goal: bool
    trajectory: list[tuple[int, int]]
    visit_counts: dict[tuple[int, int], int]


def _base_heatmap(
    visit_counts: dict[tuple[int, int], int],
    rows: int,
    cols: int,
    stat="Visit Count",
    folder: str = "output",
    filename: str | None = None,
    scale_max: int = 20,
    grid: list[list[Cell]] | None = None,
    q_table: dict[tuple[int, int], dict[str, float]] | None = None,
    show: bool = True,
) -> str:
    numpy_map = np.zeros((rows, cols))
    for (x, y), count in visit_counts.items():
        if grid:
            cell = grid[x][y]
            if cell.obstacle:
                numpy_map[x, y] = np.nan
                continue
        numpy_map[x, y] = count

    name = f"Gridworld {stat} Heatmap"
    cmap = plt.cm.Oranges
    cmap.set_bad(color="black")
    plt.imshow(
        numpy_map,
        cmap=cmap,
        interpolation="nearest",
        vmax=scale_max,
        extent=[0, cols, rows, 0],
    )

    if q_table:
        all_confidences = [
            max(actions.values()) - sorted(actions.values())[-2]
            for actions in q_table.values()
            if len(set(actions.values())) > 1
        ]
        max_conf = max(all_confidences) if all_confidences else 1.0
        if not max_conf:
            max_conf = 1.0
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

            cx, cy = y + 0.5, x + 0.5  # center of the cell

            if UP in max_actions:
                plt.arrow(
                    cx,
                    cy,
                    0,
                    -scale,
                    head_width=0.1 * scale,
                    head_length=0.2 * scale,
                    fc=color,
                    ec=color,
                )
            elif DOWN in max_actions:
                plt.arrow(
                    cx,
                    cy,
                    0,
                    scale,
                    head_width=0.1 * scale,
                    head_length=0.2 * scale,
                    fc=color,
                    ec=color,
                )
            elif LEFT in max_actions:
                plt.arrow(
                    cx,
                    cy,
                    -scale,
                    0,
                    head_width=0.1 * scale,
                    head_length=0.2 * scale,
                    fc=color,
                    ec=color,
                )
            elif RIGHT in max_actions:
                plt.arrow(
                    cx,
                    cy,
                    scale,
                    0,
                    head_width=0.1 * scale,
                    head_length=0.2 * scale,
                    fc=color,
                    ec=color,
                )

    if grid:
        for r in range(rows):
            for c in range(cols):
                cell = grid[r][c]
                x, y = c, r  # note: matplotlib has (x=cols, y=rows)

                if cell.walls and cell.walls.up:
                    plt.plot([x, x + 1], [y, y], color="black", linewidth=2)
                if cell.walls and cell.walls.down:
                    plt.plot([x, x + 1], [y + 1, y + 1], color="black", linewidth=2)
                if cell.walls and cell.walls.left:
                    plt.plot([x, x], [y, y + 1], color="black", linewidth=2)
                if cell.walls and cell.walls.right:
                    plt.plot([x + 1, x + 1], [y, y + 1], color="black", linewidth=2)
    plt.colorbar(label=stat)
    plt.title(name)
    if show:
        plt.show(block=False)
        plt.pause(2)
    filename = (
        filename
        or f'{name.lower().replace(" ", "_").replace("(", "_").replace(")", "_")}'
    )
    plt.savefig(f"{folder}/{filename}.png")
    plt.close()
    return f"{filename}.png"


def render_heatmap(
    *,
    visit_counts: dict[tuple[int, int], int],
    rows: int,
    cols: int,
    stat="Visit Count",
    folder: str = "output",
    filename: str | None = None,
    scale_max: int = 20,
    grid: list[list[Cell]] | None = None,
    q_table: dict[tuple[int, int], dict[str, float]] | None = None,
    show: bool = True,
) -> str:
    return _base_heatmap(
        visit_counts=visit_counts,
        rows=rows,
        cols=cols,
        stat=stat,
        folder=folder,
        filename=filename,
        scale_max=scale_max,
        grid=grid,
        q_table=q_table,
        show=show,
    )


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
    if not max_conf:
        max_conf = 1.0

    numpy_map = np.zeros((rows, cols))
    for (x, y), count in visit_counts.items():
        numpy_map[x, y] = count

    name = f"Gridworld {stat} Heatmap"
    plt.imshow(numpy_map, cmap="Blues", interpolation="nearest", vmax=scale_max)
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
    filename = (
        filename
        or f'{name.lower().replace(" ", "_").replace("(", "_").replace(")", "_")}'
    )
    file_location = f"{folder}/{filename}.png"
    plt.savefig(file_location)
    plt.close()
    return file_location


def line_plot(
    x_values: list[int],
    y_values: dict[str, list[float]],
    title: str,
    x_label: str,
    folder: str = "output",
    filename: str | None = None,
) -> None:
    for label, y in y_values.items():
        plt.plot(x_values, y, label=label)
    plt.title(title)
    plt.xlabel(x_label)
    plt.grid()
    plt.legend()
    filename = filename or f"{title.lower().replace(' ', '_')}"
    plt.savefig(f"{folder}/{filename}.png")
    plt.close()
