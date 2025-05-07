from functools import reduce
from os import mkdir
import os
import shutil
from gridworld.agents.generic_agent import Agent
from gridworld.agents.q_learning_agent import QLearningAgent
from gridworld.components.grid_environment import GridWorldEnv, VisitCounter
from rich.console import Console
from gridworld.utils import render_directional_heatmap_for_q_table, render_heatmap

from gridworld.runner import Runner

console = Console()

folder = f"output/gridworld-learing-history"
output_file = f"{folder}/output.txt"

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


def run_test(env: GridWorldEnv, agent: Agent, iteration: int, render: bool = False):
    runner = Runner(env, agent)
    results = runner.run_episodes(3, render=False)
    analysis = runner.analyze_results(results)

    visit_counts = [result["visit_counts"] for result in results]
    avg_visit_counts = VisitCounter.avg(*visit_counts)

    log(f"Analysis of 10 episodes with {agent.__class__.__name__} - {iteration}:")
    log("Average Reward:", analysis["reward"]["average"])
    log("Max Reward:", analysis["reward"]["max"])
    log("Min Reward:", analysis["reward"]["min"])
    log("Average Steps:", analysis["steps"]["average"])
    log("Max Steps:", analysis["steps"]["max"])
    log("Min Steps:", analysis["steps"]["min"])
    log("Reached Goal Count:", analysis["reached_goal"]["count"])
    log("Reached Goal Percentage:", analysis["reached_goal"]["count"] / 10 * 100)
    render_directional_heatmap_for_q_table(
        visit_counts=avg_visit_counts,
        rows=env.rows,
        cols=env.cols,
        q_table=agent.q_table,
        stat=f"Avg Directional Visit Count ({agent.__class__.__name__})",
        folder=folder,
        scale_max=5,
        filename=f"avg_directional_visit_count_{iteration}",
    )

    log()


env = GridWorldEnv()
learning_agent = QLearningAgent()
for i in range(15):
    run_test(env, learning_agent, iteration=i + 1, render=False)
