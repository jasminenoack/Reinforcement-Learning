from datetime import datetime
from functools import reduce
from os import mkdir
from gridworld.agents.manhattan_agent import ManhattanAgent
from gridworld.agents.generic_agent import Agent
from gridworld.agents.q_learning_agent import QLearningAgent
from gridworld.agents.random_agent import RandomAgent
from gridworld.components.grid_environment import GridWorldEnv, VisitCounter
from rich.console import Console
from gridworld.utils import render_heatmap

from gridworld.runner import Runner

console = Console()

folder = f"output/gridworld-{datetime.now().isoformat()}"
output_file = f"{folder}/output.txt"
try:
    mkdir(folder)
except FileExistsError:
    pass


def log(*message: list[str]):
    console.print(*message)
    with open(output_file, "a") as f:
        f.write(" ".join(str(m) for m in message) + "\n")


def run_test(env: GridWorldEnv, agent: Agent, render: bool = False):
    runner = Runner(env, agent)
    results = runner.run_episodes(10, render=False)
    analysis = runner.analyze_results(results)

    visit_counts = [result["visit_counts"] for result in results]
    avg_visit_counts = VisitCounter.avg(*visit_counts)
    total_visit_counts = reduce(lambda x, y: x + y, visit_counts, VisitCounter())

    log(f"Analysis of 10 episodes with {agent.__class__.__name__}:")
    log("Average Reward:", analysis["reward"]["average"])
    log("Max Reward:", analysis["reward"]["max"])
    log("Min Reward:", analysis["reward"]["min"])
    log("Average Steps:", analysis["steps"]["average"])
    log("Max Steps:", analysis["steps"]["max"])
    log("Min Steps:", analysis["steps"]["min"])
    log("Reached Goal Count:", analysis["reached_goal"]["count"])
    log("Reached Goal Percentage:", analysis["reached_goal"]["count"] / 10 * 100)
    render_heatmap(
        visit_counts=avg_visit_counts,
        rows=env.rows,
        cols=env.cols,
        stat=f"Avg Visit Count ({agent.__class__.__name__})",
        folder=folder,
        scale_max=5,
    )
    render_heatmap(
        visit_counts=total_visit_counts,
        rows=env.rows,
        cols=env.cols,
        stat=f"Total Visit Count ({agent.__class__.__name__})",
        folder=folder,
        scale_max=25,
    )
    log()


random_agent = RandomAgent()
env = GridWorldEnv()
run_test(env, random_agent, render=False)
designed_agent = ManhattanAgent()
run_test(env, designed_agent, render=False)
learning_agent = QLearningAgent()
run_test(env, learning_agent, render=False)
