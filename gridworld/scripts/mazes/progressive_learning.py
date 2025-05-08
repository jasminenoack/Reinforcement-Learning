from dataclasses import dataclass
from os import mkdir
import os
import shutil
from gridworld.agents.generic_agent import Agent
from gridworld.agents.q_learning_agent import QLearningAgent
from gridworld.components.grid_environment import GridWorldEnv, VisitCounter
from rich.console import Console
from gridworld.components.maze_builders import (
    RecursiveBacktracking,
    SparseObstacleMazeGenerator,
)
from gridworld.utils import line_plot, render_heatmap

from gridworld.runner import Runner

console = Console()


folder = f"output/progressive"
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


PER_ITERATION = 5
NUMBER_OF_ITERATIONS = 10
SIZE = 10


@dataclass
class Summary:
    current_iterations: int
    avg_reward: float
    avg_steps: float
    reached_goal_percentage: float


def run_test(env: GridWorldEnv, agent: Agent, render: bool = False, prefix: str = ""):
    runner = Runner(env, agent)
    results = runner.run_episodes(PER_ITERATION, render=False)
    analysis = runner.analyze_results(results)

    visit_counts = [result["visit_counts"] for result in results]
    avg_visit_counts = VisitCounter.avg(*visit_counts)

    log(f"Analysis of {len(results)} episodes with {agent.__class__.__name__}:")
    log("Average Reward:", analysis["reward"]["average"])
    log("Average Steps:", analysis["steps"]["average"])
    log(
        "Reached Goal Percentage:",
        analysis["reached_goal"]["count"] / len(results) * 100,
    )
    filename = render_heatmap(
        visit_counts=avg_visit_counts,
        rows=env.rows,
        cols=env.cols,
        stat=f"{prefix} Avg Visit Count ({agent.__class__.__name__})",
        folder=folder,
        scale_max=1,
        grid=env.grid,
        q_table=agent.q_table,
        show=False,
    )
    log(f"![{filename}](./{filename})")
    return Summary(
        current_iterations=len(results),
        avg_reward=analysis["reward"]["average"],
        avg_steps=analysis["steps"]["average"],
        reached_goal_percentage=analysis["reached_goal"]["count"] / len(results) * 100,
    )


def write_summary_table(summaries: list[Summary]) -> None:
    current_iterations = 0
    log("| Iter | Avg Reward | Avg Steps |  Goal % |")
    log("|------|-------------|-----------|--------|")
    for s in summaries:
        current_iterations += s.current_iterations
        log(
            f"| {current_iterations - 2} - {current_iterations} "
            f"| {s.avg_reward:.2f} "
            f"| {s.avg_steps:.2f} "
            f"| {s.reached_goal_percentage:.1f}% |"
        )
    log()


def write_summary_charts(summaries: list[Summary], prefix: str) -> None:
    x_values = []
    for s in summaries:
        previous_iterations = 0 if not x_values else x_values[-1]
        x_values.append(previous_iterations + s.current_iterations)
    avg_rewards = [s.avg_reward for s in summaries]
    line_plot(
        x_values=x_values,
        y_values={
            "Avg Reward": avg_rewards,
        },
        title="Avg Reward",
        x_label="Iterations",
        folder=folder,
        filename=f"{prefix}-avg_reward",
    )
    log(f"![avg_reward](./{prefix}-avg_reward.png)")
    line_plot(
        x_values=x_values,
        y_values={
            "Avg Steps": [s.avg_steps for s in summaries],
        },
        title="Avg Steps",
        x_label="Iterations",
        folder=folder,
        filename=f"{prefix}-avg_steps",
    )
    log(f"![avg_steps](./{prefix}-avg_steps.png)")
    line_plot(
        x_values=x_values,
        y_values={
            "Reached Goal %": [s.reached_goal_percentage for s in summaries],
        },
        title="Reached Goal",
        x_label="Iterations",
        folder=folder,
        filename=f"{prefix}-reached_goal",
    )
    log(f"![reached_goal](./{prefix}-reached_goal.png)")


def write_multiple_summary_charts(
    summaries: dict[str, list[Summary]],
) -> None:
    x_values = []
    for summary in list(summaries.values())[0]:
        previous_iterations = 0 if not x_values else x_values[-1]
        x_values.append(previous_iterations + summary.current_iterations)

    avg_reward_y_values = {}
    avg_steps_y_values = {}
    reached_goal_y_values = {}
    for key, summary in summaries.items():
        avg_reward_y_values[key] = [s.avg_reward for s in summary]
        avg_steps_y_values[key] = [s.avg_steps for s in summary]
        reached_goal_y_values[key] = [s.reached_goal_percentage for s in summary]

    line_plot(
        x_values=x_values,
        y_values=avg_reward_y_values,
        title="Avg Reward",
        x_label="Iterations",
        folder=folder,
        filename="all_avg_reward",
    )
    log(f"![all_avg_reward](./all_avg_reward.png)")

    line_plot(
        x_values=x_values,
        y_values=avg_steps_y_values,
        title="Avg Steps",
        x_label="Iterations",
        folder=folder,
        filename="all_avg_steps",
    )
    log(f"![all_avg_steps](./all_avg_steps.png)")

    line_plot(
        x_values=x_values,
        y_values=reached_goal_y_values,
        title="Reached Goal",
        x_label="Iterations",
        folder=folder,
        filename="all_reached_goal",
    )
    log(f"![all_reached_goal](./all_reached_goal.png)")


OF_ENV = 3

agent = QLearningAgent()

empty_envs = [
    GridWorldEnv(
        max_steps=100,
    )
    for _ in range(OF_ENV)
]

sparse_envs = [
    GridWorldEnv(
        grid=SparseObstacleMazeGenerator(
            rows=SIZE,
            cols=SIZE,
        ).run(),
        max_steps=100,
    )
    for _ in range(OF_ENV)
]

maze_envs = [
    GridWorldEnv(
        grid=RecursiveBacktracking(
            rows=SIZE,
            cols=SIZE,
        ).run(),
        max_steps=100,
    )
    for _ in range(OF_ENV)
]

envs = [*empty_envs, *sparse_envs, *maze_envs]

all_summaries = {}

for env_num, env in enumerate(envs):
    summaries = []
    for i in range(NUMBER_OF_ITERATIONS):
        summaries.append(
            run_test(
                env,
                agent,
                render=False,
                prefix=f"{i+1}-{env.__class__.__name__}-{env_num}-",
            )
        )
    write_summary_table(summaries)
    write_summary_charts(summaries, prefix=f"{env.__class__.__name__}-{env_num}")
    all_summaries[f"{env.__class__.__name__}-{env_num}"] = summaries


console.print()
console.print("All summaries:")
console.print()
write_multiple_summary_charts(all_summaries)
