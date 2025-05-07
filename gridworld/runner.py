import time
from gridworld.agents.generic_agent import Agent
from gridworld.agents.random_agent import RandomAgent
from gridworld.components.grid_environment import GridWorldEnv
from rich.console import Console

from gridworld.utils import RunnerReturn, Step


console = Console()


class Runner:
    def __init__(self, env: GridWorldEnv, agent: Agent):
        self.env = env
        self.agent = agent

    def run_episode(
        self,
        *,
        render: bool = False,
        clear_render: bool = True,
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
            step_result = self.env.step(action)
            new_state, reward, done = step_result
            step_result = Step(
                start=state,
                action=action,
                reward=reward,
                new_state=new_state,
                done=done,
            )
            self.agent.observe(step_result)
            trajectory.append(
                Step(
                    start=state,
                    action=action,
                    reward=reward,
                    new_state=new_state,
                    done=done,
                )
            )
            state = new_state
            if render:
                time.sleep(sleep)

        return RunnerReturn(
            total_reward=self.env.total_reward,
            steps=len(trajectory),
            reached_goal=self.env.done,
            trajectory=trajectory,
        )

    def run_episodes(
        self,
        num_episodes: int,
        render: bool = False,
        clear_render: bool = True,
        sleep: float = 0.5,
    ) -> list[RunnerReturn]:
        results = []
        for _ in range(num_episodes):
            result = self.run_episode(
                render=render, clear_render=clear_render, sleep=sleep
            )
            results.append(result)
        return results

    def analyze_results(self, results: list[RunnerReturn]) -> dict:
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
    agent = RandomAgent()
    env = GridWorldEnv()
    runner = Runner(env, agent)
    results = runner.run_episodes(10, render=False)
    analysis = runner.analyze_results(results)
    console.print("Analysis of 10 episodes:")
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
