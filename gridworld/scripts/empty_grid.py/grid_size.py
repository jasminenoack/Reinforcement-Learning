from dataclasses import dataclass
from os import mkdir
import os
import shutil
from gridworld.agents.generic_agent import Agent
from gridworld.agents.q_learning_agent import QLearningAgent
from gridworld.components.grid_environment import GridWorldEnv, VisitCounter
from rich.console import Console
from gridworld.utils import (
    line_plot,
)

from gridworld.runner import Runner

console = Console()

folder = f"output/gridworld-learing-by-grid-size"
output_file = f"{folder}/output.md"

if os.path.exists(folder):
    shutil.rmtree(folder)

try:
    mkdir(folder)
except FileExistsError:
    pass


def log(*message: list[str]):
    console.print(*message)
    with open(output_file, "a") as f:
        f.write(" ".join(str(m) for m in message) + "\n")


@dataclass
class Summary:
    current_iterations: int
    avg_reward: float
    max_reward: float
    min_reward: float
    avg_steps: float
    max_steps: float
    min_steps: float
    reached_goal_count: int
    reached_goal_percentage: float


NUMBER_OF_EPISODES_PER_ITERATION = 10
TOTAL_ITERATIONS = 10


def run_test(env: GridWorldEnv, agent: Agent, iteration: int, render: bool = False):
    runner = Runner(env, agent)
    results = runner.run_episodes(NUMBER_OF_EPISODES_PER_ITERATION, render=False)
    analysis = runner.analyze_results(results)
    return Summary(
        current_iterations=len(results),
        avg_reward=analysis["reward"]["average"],
        max_reward=analysis["reward"]["max"],
        min_reward=analysis["reward"]["min"],
        avg_steps=analysis["steps"]["average"],
        max_steps=analysis["steps"]["max"],
        min_steps=analysis["steps"]["min"],
        reached_goal_count=analysis["reached_goal"]["count"],
        reached_goal_percentage=analysis["reached_goal"]["count"] / 3 * 100,
    )


def write_summary_table(summaries: list[Summary]) -> None:
    current_iterations = 0
    log(
        "| Iter | Avg Reward | Max Reward | Min Reward | Avg Steps | Max Steps | Min Steps | Reached Goal | Goal % |"
    )
    log(
        "|------|-------------|------------|------------|-----------|-----------|-----------|---------------|--------|"
    )
    for s in summaries:
        current_iterations += s.current_iterations
        log(
            f"| {current_iterations - 2} - {current_iterations} "
            f"| {s.avg_reward:.2f} "
            f"| {s.max_reward:.2f} "
            f"| {s.min_reward:.2f} "
            f"| {s.avg_steps:.2f} "
            f"| {s.max_steps:.0f} "
            f"| {s.min_steps:.0f} "
            f"| {s.reached_goal_count} "
            f"| {s.reached_goal_percentage:.1f}% |"
        )
    log()


def write_summary_charts(summaries: list[Summary], grid_size) -> None:
    x_values = []
    for s in summaries:
        previous_iterations = 0 if not x_values else x_values[-1]
        x_values.append(previous_iterations + s.current_iterations)
    avg_rewards = [s.avg_reward for s in summaries]
    max_rewards = [s.max_reward for s in summaries]
    min_rewards = [s.min_reward for s in summaries]
    line_plot(
        x_values=x_values,
        y_values={
            "Avg Reward": avg_rewards,
            "Max Reward": max_rewards,
            "Min Reward": min_rewards,
        },
        title="Avg Reward",
        x_label="Iterations",
        folder=folder,
        filename=f"avg_reward-{grid_size}",
    )
    log(f"![avg_reward](./avg_reward-{grid_size}.png)")
    line_plot(
        x_values=x_values,
        y_values={
            "Avg Steps": [s.avg_steps for s in summaries],
            "Max Steps": [s.max_steps for s in summaries],
            "Min Steps": [s.min_steps for s in summaries],
        },
        title="Avg Steps",
        x_label="Iterations",
        folder=folder,
        filename=f"avg_steps-{grid_size}",
    )
    log(f"![avg_steps](./avg_steps-{grid_size}.png)")
    line_plot(
        x_values=x_values,
        y_values={
            "Reached Goal": [s.reached_goal_count for s in summaries],
            "Reached Goal %": [s.reached_goal_percentage for s in summaries],
        },
        title="Reached Goal",
        x_label="Iterations",
        folder=folder,
        filename=f"reached_goal-{grid_size}",
    )
    log(f"![reached_goal](./reached_goal-{grid_size}.png)")


def run_env(rows: int, cols: int) -> list[Summary]:
    log(f"Running env with grid_size: {rows}x{cols}")
    summaries = []
    env = GridWorldEnv(rows=rows, cols=cols)
    agent = QLearningAgent()
    for i in range(10):
        summaries.append(run_test(env, agent, iteration=i, render=False))
    write_summary_table(
        summaries=summaries,
    )
    write_summary_charts(
        summaries=summaries,
        grid_size=f"{rows}x{cols}",
    )
    return summaries


def cross_summary_table(
    summary_by_grid_sizes: dict[int, list[Summary]],
) -> None:
    log("| Total Iterations | Steps | Avg Reward | Avg Steps | Reached Goal | Goal % |")
    log("|------------------|-------|------------|-----------|--------------|--------|")
    for grid_size in summary_by_grid_sizes:
        total_iterations = sum(
            summary.current_iterations for summary in summary_by_grid_sizes[grid_size]
        )
        avg_reward = sum(
            summary.avg_reward for summary in summary_by_grid_sizes[grid_size]
        ) / len(summary_by_grid_sizes[grid_size])
        avg_steps = sum(
            summary.avg_steps for summary in summary_by_grid_sizes[grid_size]
        ) / len(summary_by_grid_sizes[grid_size])
        reached_goal = sum(
            summary.reached_goal_count for summary in summary_by_grid_sizes[grid_size]
        )
        reached_goal_percentage = reached_goal / total_iterations * 100
        log(
            f"| {total_iterations} | {grid_size} | {avg_reward:.2f} | {avg_steps:.2f} | "
            f"{reached_goal} | {reached_goal_percentage:.1f}% |"
        )
    log()


def write_cross_summary_charts(
    summary_by_grid_sizes: dict[int, list[Summary]],
) -> None:
    x_values = list(
        range(
            NUMBER_OF_EPISODES_PER_ITERATION,
            TOTAL_ITERATIONS * NUMBER_OF_EPISODES_PER_ITERATION + 1,
            NUMBER_OF_EPISODES_PER_ITERATION,
        )
    )
    grid_sizes = list(summary_by_grid_sizes.keys())

    y_values_rewards = {}
    y_values_steps = {}
    y_values_reached_goal = {}

    for grid_size in grid_sizes:
        y_values_rewards[f"{grid_size} grid"] = [
            summary.avg_reward for summary in summary_by_grid_sizes[grid_size]
        ]
        y_values_steps[grid_size] = [
            summary.avg_steps for summary in summary_by_grid_sizes[grid_size]
        ]
        y_values_reached_goal[grid_size] = [
            summary.reached_goal_count for summary in summary_by_grid_sizes[grid_size]
        ]

    line_plot(
        x_values=x_values,
        y_values=y_values_rewards,
        title="Avg Reward by grid size",
        x_label="Iterations",
        folder=folder,
        filename=f"avg_reward",
    )
    log(f"![avg_reward](./avg_reward.png)")

    line_plot(
        x_values=x_values,
        y_values=y_values_steps,
        title="Avg Steps by grid size",
        x_label="Iterations",
        folder=folder,
        filename=f"avg_steps",
    )
    log(f"![avg_steps](./avg_steps.png)")

    line_plot(
        x_values=x_values,
        y_values=y_values_reached_goal,
        title="Reached Goal by grid size",
        x_label="Iterations",
        folder=folder,
        filename=f"reached_goal",
    )
    log(f"![reached_goal](./reached_goal.png)")


summaries_by_grid_size = {
    5_5: run_env(rows=5, cols=5),
    7_7: run_env(rows=7, cols=7),
    10_10: run_env(rows=10, cols=10),
    12_12: run_env(rows=12, cols=12),
    15_15: run_env(rows=15, cols=15),
    20_20: run_env(rows=20, cols=20),
    25_25: run_env(rows=25, cols=25),
}

cross_summary_table(
    summary_by_grid_sizes=summaries_by_grid_size,
)
write_cross_summary_charts(
    summary_by_grid_sizes=summaries_by_grid_size,
)
