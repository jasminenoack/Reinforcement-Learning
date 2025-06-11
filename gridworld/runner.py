import os
import shutil
import time
from gridworld.agents.manhattan_agent import ManhattanAgent
from gridworld.agents.generic_agent import Agent
from gridworld.components.grid_environment import GridWorldEnv, StepResult
from rich.console import Console

from gridworld.components.maze_builders import (
    RecursiveBacktracking,
    SparseObstacleMazeGenerator,
)
from gridworld.utils import RunnerReturn, Step, render_heatmap


console = Console()


class Runner:
    def __init__(self, env: GridWorldEnv, agent: Agent):
        self.env = env
        self.agent = agent

    def run_episode(
        self,
        *,
        render: bool = False,
        clear_render: bool = False,
        sleep: float = 0.5,
    ) -> RunnerReturn:
        trajectory: list[Step] = []
        # make sure the environment is in a clean state
        self.env.reset()
        self.agent.reset()
        done = False
        state = self.env.get_state()

        while not self.env.done:
            if clear_render:
                console.clear()
            if render:
                self.env.render()

            action = self.agent.act(state)
            step_data: StepResult = self.env.step(action)
            new_state = step_data.new_state
            reward = step_data.reward
            done = step_data.done

            step = Step(
                start=state,
                action=action,
                reward=reward,
                new_state=new_state,
                done=done,
            )
            self.agent.observe(step)
            trajectory.append(step)
            state = new_state
            if render:
                time.sleep(sleep)
        if render:
            if clear_render:
                console.clear()
            self.env.render()
            console.print(
                f"Total reward: {self.env.total_reward}, Steps: {len(trajectory)}"
            )
            console.print(f"Visit counts: {self.env.visit_counts}")
            render_heatmap(
                visit_counts=self.env.visit_counts.data,
                rows=self.env.rows,
                cols=self.env.cols,
                scale_max=3,
            )

        return RunnerReturn(
            total_reward=self.env.total_reward,
            steps=len(trajectory),
            reached_goal=self.env.reached_goal,
            trajectory=trajectory,
            visit_counts=self.env.visit_counts,
        )

    def run_episodes(
        self,
        num_episodes: int,
        render: bool = False,
        clear_render: bool = False,
        sleep: float = 0.5,
    ) -> list[RunnerReturn]:
        results = []
        for _ in range(num_episodes):
            result = self.run_episode(
                render=render, clear_render=clear_render, sleep=sleep
            )
            results.append(result)
        return results

    def analyze_results(self, results: list[RunnerReturn]) -> dict[str, dict[str, float]]:
        total_rewards = [result["total_reward"] for result in results]
        average_reward = sum(total_rewards) / len(total_rewards)

        return {
            "reward": {
                "average": average_reward,
                "max": max(total_rewards),
                "min": min(total_rewards),
            },
            "steps": {
                "average": sum(result["steps"] for result in results) / len(results),
                "max": max(result["steps"] for result in results),
                "min": min(result["steps"] for result in results),
            },
            "reached_goal": {
                "count": sum(result["reached_goal"] for result in results),
            },
        }


if __name__ == "__main__":
    folder = f"output/runner-main"
    output_file = f"{folder}/output.txt"
    if os.path.exists(folder):
        shutil.rmtree(folder)

    try:
        os.mkdir(folder)
    except FileExistsError:
        pass

    obstacle_maze = SparseObstacleMazeGenerator(
        rows=10,
        cols=10,
    )
    maze = RecursiveBacktracking(
        rows=10,
        cols=10,
    )

    env = GridWorldEnv(rows=5, cols=5, grid=maze.run())
    agent = ManhattanAgent(goal=env.goal)
    runner = Runner(env, agent)
    runner.run_episode(render=True, clear_render=False, sleep=0.5)
