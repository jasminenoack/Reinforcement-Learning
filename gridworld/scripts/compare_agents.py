import time
from gridworld.agents.designed_agent import DesignedStaticMazeAgent
from gridworld.agents.generic_agent import Agent
from gridworld.agents.random_agent import RandomAgent
from gridworld.components.grid_environment import GridWorldEnv
from rich.console import Console

from gridworld.runner import Runner
from gridworld.utils import RunnerReturn, Step


console = Console()

random_agent = RandomAgent()
env = GridWorldEnv()
random_runner = Runner(env, random_agent)
random_results = random_runner.run_episodes(10, render=False)
random_analysis = random_runner.analyze_results(random_results)
console.print("Analysis of 10 episodes with random agent:")
console.print("Average Reward:", random_analysis["reward"]["average"])
console.print("Max Reward:", random_analysis["reward"]["max"])
console.print("Min Reward:", random_analysis["reward"]["min"])
console.print("Average Steps:", random_analysis["steps"]["average"])
console.print("Max Steps:", random_analysis["steps"]["max"])
console.print("Min Steps:", random_analysis["steps"]["min"])
console.print("Reached Goal Count:", random_analysis["reached_goal"]["count"])
console.print(
    "Reached Goal Percentage:", random_analysis["reached_goal"]["count"] / 10 * 100
)

console.print()
console.print()

designed_agent = DesignedStaticMazeAgent()
designed_runner = Runner(env, designed_agent)
designed_results = designed_runner.run_episodes(10, render=False)
designed_analysis = designed_runner.analyze_results(designed_results)
console.print("Analysis of 10 episodes with designed agent:")
console.print("Average Reward:", designed_analysis["reward"]["average"])
console.print("Max Reward:", designed_analysis["reward"]["max"])
console.print("Min Reward:", designed_analysis["reward"]["min"])
console.print("Average Steps:", designed_analysis["steps"]["average"])
console.print("Max Steps:", designed_analysis["steps"]["max"])
console.print("Min Steps:", designed_analysis["steps"]["min"])
console.print("Reached Goal Count:", designed_analysis["reached_goal"]["count"])
console.print(
    "Reached Goal Percentage:",
    designed_analysis["reached_goal"]["count"] / 10 * 100,
)
console.print()
