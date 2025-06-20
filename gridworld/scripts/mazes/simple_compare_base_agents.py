from os import mkdir
import os
import shutil
from typing import Any
from gridworld.agents.lost_agent import LostAgent
from gridworld.agents.manhattan_agent import ManhattanAgent
from gridworld.agents.generic_agent import Agent
from gridworld.agents.q_learning_agent import QLearningAgent
from gridworld.agents.random_agent import RandomAgent
from gridworld.components.grid_environment import GridWorldEnv, VisitCounter
from rich.console import Console
from gridworld.components.maze_builders import RecursiveBacktracking
from gridworld.utils import render_heatmap

from gridworld.runner import Runner

console = Console()

folder = "output/gridworld-simple-comparison"
output_file = f"{folder}/output.txt"
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


NUMBER_OF_ITERATIONS = 10


def run_test(env: GridWorldEnv, agent: Agent, render: bool = False):
    runner = Runner(env, agent)
    results = runner.run_episodes(NUMBER_OF_ITERATIONS, render=False)
    analysis = runner.analyze_results(results)

    visit_counts = [result["visit_counts"] for result in results]
    avg_visit_counts = VisitCounter.avg(*visit_counts)

    log(f"Analysis of 10 episodes with {agent.__class__.__name__}:")
    log("Average Reward:", analysis["reward"]["average"])
    log("Max Reward:", analysis["reward"]["max"])
    log("Min Reward:", analysis["reward"]["min"])
    log("Average Steps:", analysis["steps"]["average"])
    log("Max Steps:", analysis["steps"]["max"])
    log("Min Steps:", analysis["steps"]["min"])
    log("Reached Goal Count:", analysis["reached_goal"]["count"])
    log(
        "Reached Goal Percentage:",
        analysis["reached_goal"]["count"] / len(results) * 100,
    )
    render_heatmap(
        visit_counts=avg_visit_counts.data,
        rows=env.rows,
        cols=env.cols,
        stat=f"Avg Visit Count ({agent.__class__.__name__})",
        folder=folder,
        scale_max=1,
        grid=env.grid,
    )
    log()


agents = [
    RandomAgent(),
    LostAgent(),
    ManhattanAgent(),
    QLearningAgent(),
]

grid = RecursiveBacktracking(
    rows=5,
    cols=5,
)
env = GridWorldEnv(grid=grid.run(), max_steps=100)

for agent in agents:
    run_test(env, agent, render=False)
