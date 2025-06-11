from dataclasses import dataclass
from os import mkdir
import os
import shutil
from typing import Any
from gridworld.agents.generic_agent import Agent
from gridworld.agents.q_learning_agent import QLearningAgent
from gridworld.components.grid_environment import GridWorldEnv
from rich.console import Console
from gridworld.utils import (
    line_plot,
)

from gridworld.runner import Runner

console = Console()

folder = f"output/gridworld-learning-by-steps"
output_file = f"{folder}/output.md"

if os.path.exists(folder):
    shutil.rmtree(folder)

try:
    mkdir(folder)
except FileExistsError:
    pass


def log(*message: Any) -> None:
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


def write_summary_table(summaries: list[Summary], steps: int) -> None:
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


def write_summary_charts(summaries: list[Summary], steps: int) -> None:
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
        filename=f"avg_reward-{steps}",
    )
    log(f"![avg_reward](./avg_reward-{steps}.png)")
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
        filename=f"avg_steps-{steps}",
    )
    log(f"![avg_steps](./avg_steps-{steps}.png)")
    line_plot(
        x_values=x_values,
        y_values={
            "Reached Goal": [s.reached_goal_count for s in summaries],
            "Reached Goal %": [s.reached_goal_percentage for s in summaries],
        },
        title="Reached Goal",
        x_label="Iterations",
        folder=folder,
        filename=f"reached_goal-{steps}",
    )
    log(f"![reached_goal](./reached_goal-{steps}.png)")


def run_env(max_steps: int) -> list[Summary]:
    log(f"Running env with max_steps: {max_steps}")
    summaries = []
    env = GridWorldEnv(max_steps=max_steps)
    agent = QLearningAgent()
    for i in range(10):
        summaries.append(run_test(env, agent, iteration=i, render=False))
    write_summary_table(
        summaries=summaries,
        steps=max_steps,
    )
    write_summary_charts(
        summaries=summaries,
        steps=max_steps,
    )
    return summaries


def cross_summary_table(
    summaries_by_steps: dict[int, list[Summary]],
    steps: list[int],
) -> None:
    log("| Total Iterations | Steps | Avg Reward | Avg Steps | Reached Goal | Goal % |")
    log("|------------------|-------|------------|-----------|--------------|--------|")
    for step in steps:
        total_iterations = sum(
            summary.current_iterations for summary in summaries_by_steps[step]
        )
        avg_reward = sum(
            summary.avg_reward for summary in summaries_by_steps[step]
        ) / len(summaries_by_steps[step])
        avg_steps = sum(
            summary.avg_steps for summary in summaries_by_steps[step]
        ) / len(summaries_by_steps[step])
        reached_goal = sum(
            summary.reached_goal_count for summary in summaries_by_steps[step]
        )
        reached_goal_percentage = reached_goal / total_iterations * 100
        log(
            f"| {total_iterations} | {step} | {avg_reward:.2f} | {avg_steps:.2f} | "
            f"{reached_goal} | {reached_goal_percentage:.1f}% |"
        )
        log()


def write_cross_summary_charts(
    summaries_by_steps: dict[int, list[Summary]],
    steps: list[int],
) -> None:
    x_values = list(
        range(
            NUMBER_OF_EPISODES_PER_ITERATION,
            TOTAL_ITERATIONS * NUMBER_OF_EPISODES_PER_ITERATION + 1,
            NUMBER_OF_EPISODES_PER_ITERATION,
        )
    )

    y_values_rewards = {}
    y_values_steps = {}
    y_values_reached_goal = {}

    for step in steps:
        y_values_rewards[f"{step} steps"] = [
            summary.avg_reward for summary in summaries_by_steps[step]
        ]
        y_values_steps[step] = [
            summary.avg_steps for summary in summaries_by_steps[step]
        ]
        y_values_reached_goal[step] = [
            summary.reached_goal_count for summary in summaries_by_steps[step]
        ]

    line_plot(
        x_values=x_values,
        y_values=y_values_rewards,
        title="Avg Reward by allowed steps",
        x_label="Iterations",
        folder=folder,
        filename=f"avg_reward",
    )
    log(f"![avg_reward](./avg_reward.png)")

    line_plot(
        x_values=x_values,
        y_values=y_values_steps,
        title="Avg Steps by allowed steps",
        x_label="Iterations",
        folder=folder,
        filename=f"avg_steps",
    )
    log(f"![avg_steps](./avg_steps.png)")

    line_plot(
        x_values=x_values,
        y_values=y_values_reached_goal,
        title="Reached Goal by allowed steps",
        x_label="Iterations",
        folder=folder,
        filename=f"reached_goal",
    )
    log(f"![reached_goal](./reached_goal.png)")


summaries_by_steps = {
    10: run_env(10),
    20: run_env(20),
    30: run_env(30),
    40: run_env(40),
    50: run_env(50),
    60: run_env(60),
    70: run_env(70),
    80: run_env(80),
    90: run_env(90),
    100: run_env(100),
    110: run_env(110),
    120: run_env(120),
    130: run_env(130),
    140: run_env(140),
    150: run_env(150),
}

cross_summary_table(
    summaries_by_steps=summaries_by_steps,
    steps=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
)
write_cross_summary_charts(
    summaries_by_steps=summaries_by_steps,
    steps=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
)
