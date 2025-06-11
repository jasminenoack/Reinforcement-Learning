from dataclasses import dataclass
from functools import reduce
from os import mkdir
import os
import shutil
from typing import Any
from gridworld.agents.generic_agent import Agent
from gridworld.agents.q_learning_agent import QLearningAgent
from gridworld.components.grid_environment import GridWorldEnv, VisitCounter
from rich.console import Console
from gridworld.utils import (
    line_plot,
    render_directional_heatmap_for_q_table,
)

from gridworld.runner import Runner

console = Console()

folder = f"output/gridworld-learning-history"
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


summaries = []

NUMBER_OF_EPISODES_PER_ITERATION = 30


def run_test(env: GridWorldEnv, agent: QLearningAgent, iteration: int, render: bool = False) -> None:
    runner = Runner(env, agent)
    results = runner.run_episodes(NUMBER_OF_EPISODES_PER_ITERATION, render=False)
    analysis = runner.analyze_results(results)

    visit_counts = [result["visit_counts"] for result in results]
    avg_visit_counts = VisitCounter.avg(*visit_counts)

    summaries.append(
        Summary(
            current_iterations=len(results),
            avg_reward=analysis["reward"]["average"],
            max_reward=analysis["reward"]["max"],
            min_reward=analysis["reward"]["min"],
            avg_steps=analysis["steps"]["average"],
            max_steps=analysis["steps"]["max"],
            min_steps=analysis["steps"]["min"],
            reached_goal_count=analysis["reached_goal"]["count"],
            reached_goal_percentage=analysis["reached_goal"]["count"]
            / len(results)
            * 100,
        )
    )

    file_name = f"avg_directional_visit_count_{iteration}"
    render_directional_heatmap_for_q_table(
        visit_counts=avg_visit_counts.data,
        rows=env.rows,
        cols=env.cols,
        q_table=agent.q_table,
        stat=f"Avg Directional Visit Count ({agent.__class__.__name__})",
        folder=folder,
        scale_max=5,
        filename=file_name,
    )
    log(f"![{file_name}](./{file_name}.png)")


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


def write_summary_charts(summaries: list[Summary]) -> None:
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
        filename="avg_reward",
    )
    log(f"![avg_reward](./avg_reward.png)")
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
        filename="avg_steps",
    )
    log(f"![avg_steps](./avg_steps.png)")
    line_plot(
        x_values=x_values,
        y_values={
            "Reached Goal": [s.reached_goal_count for s in summaries],
            "Reached Goal %": [s.reached_goal_percentage for s in summaries],
        },
        title="Reached Goal",
        x_label="Iterations",
        folder=folder,
        filename="reached_goal",
    )
    log(f"![reached_goal](./reached_goal.png)")


env = GridWorldEnv(rows=10, cols=10)
env.max_steps = 100
learning_agent = QLearningAgent()
log(
    "Learning Agent:",
    learning_agent.__class__.__name__,
    "with",
    learning_agent.epsilon,
    "epsilon and",
    learning_agent.alpha,
    "alpha.",
)
for i in range(15):
    run_test(env, learning_agent, iteration=i, render=False)


log("Learning Agent Summary:")
write_summary_table(
    summaries=summaries,
)
write_summary_charts(
    summaries=summaries,
)
