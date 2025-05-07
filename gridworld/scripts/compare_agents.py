import time
from gridworld.agents.designed_agent import DesignedStaticMazeAgent
from gridworld.agents.generic_agent import Agent
from gridworld.agents.random_agent import RandomAgent
from gridworld.components.grid_environment import GridWorldEnv
from rich.console import Console

from gridworld.runner import Runner
from gridworld.utils import RunnerReturn, Step

console = Console()


def run_test(env: GridWorldEnv, agent: Agent, render: bool = False):
    runner = Runner(env, agent)
    results = runner.run_episodes(10, render=False)
    analysis = runner.analyze_results(results)
    console.print("Analysis of 10 episodes with random agent:")
    console.print("Average Reward:", analysis["reward"]["average"])
    console.print("Max Reward:", analysis["reward"]["max"])
    console.print("Min Reward:", analysis["reward"]["min"])
    console.print("Average Steps:", analysis["steps"]["average"])
    console.print("Max Steps:", analysis["steps"]["max"])
    console.print("Min Steps:", analysis["steps"]["min"])
    console.print("Reached Goal Count:", analysis["reached_goal"]["count"])
    console.print(
        "Reached Goal Percentage:", analysis["reached_goal"]["count"] / 10 * 100
    )
    console.print()


random_agent = RandomAgent()
env = GridWorldEnv()
run_test(env, random_agent, render=False)
designed_agent = DesignedStaticMazeAgent()
run_test(env, designed_agent, render=False)
