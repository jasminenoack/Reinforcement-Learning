import os
import random
import shutil
from time import sleep
from typing import Any
from queens.agents.generic_agent import Agent
from queens.agents.reinforcement_agents import (
    DynamicEpsilonAgent,
    # SimpleAgentHighAlpha,
    # SimpleAgentHighEpsilon,
    # SimpleAgentMidAlpha,
    # SimpleAgentMidEpsilon,
    # SimpleAgentNoEpsilon,
    # SimpleRandomReinforcementAgent,
    # SimpleReinforcementAgent,
)
from queens.components.grid import (
    Grid,
    EarlyExitGrid,
)
from queens.dtos import Observation, StepResult, RunnerReturn
from queens.utils import build_board_array
import matplotlib.pyplot as plt

# from queens.agents.random_agent import RandomAgent

plt: Any

RENDER_DELAY = 0.5


class Runner:
    def __init__(self, env: Grid, agent: Agent):
        self.env = env
        self.agent = agent
        self.agent.reset()
        self.env.reset()

    def run_episode(
        self, *, render: bool = False, render_result: bool = False
    ) -> RunnerReturn:
        self.agent.reset()
        self.env.reset()
        trajectory: list[StepResult] = []
        if render:
            self.env.render()
            sleep(RENDER_DELAY)

        while not self.env.done:
            state = self.env.get_state()
            action = self.agent.act(observation=Observation(board_state=state))
            result = self.env.step(*action)
            trajectory.append(result)
            self.agent.observe_step(result)
            if render:
                self.env.render()
                sleep(RENDER_DELAY)

        if render_result:
            self.env.render()
            sleep(RENDER_DELAY)
        result = RunnerReturn(
            trajectory=trajectory,
            solved=self.env.solved,
            board=self.env.board,
            moves=self.env.moves,
            score=self.env.score,
        )
        self.agent.observe_result(result)

        return result

    def run_episodes(
        self,
        num_episodes: int,
        *,
        render: bool = False,
    ) -> list[RunnerReturn]:
        results: list[RunnerReturn] = []
        for _ in range(num_episodes):
            result = self.run_episode(render=False, render_result=render)
            results.append(result)
        return results

    def render_analytics(self, results: list[RunnerReturn]):
        solved = sum(1 for result in results if result.solved)
        failed = sum(1 for result in results if not result.solved)
        avg_moves = sum(result.moves for result in results) / len(results)
        print(f"Total solved: {solved}")
        print(f"Total failed: {failed}")
        print(f"Average moves: {avg_moves:.2f}")

    def build_heatmap(
        self, results: list[RunnerReturn], folder: str, agent_name: str, cases: int
    ):
        board_size = len(results[0].board)
        heatmap_data = [[0] * board_size for _ in range(board_size)]
        for result in results:
            for step in result.trajectory:
                row, col = step.action
                heatmap_data[row][col] += 1

        plt.imshow(
            heatmap_data, cmap="hot", interpolation="nearest", vmin=0, vmax=cases
        )
        plt.colorbar()
        plt.title("Heatmap of Actions")
        plt.xlabel("Column")
        plt.ylabel("Row")
        plt.savefig(f"{folder}/heatmap-{agent_name}.png")
        plt.show()


if __name__ == "__main__":
    size = 4
    seed = random.randint(0, 100)
    folder = f"output/queens"
    output_file = f"{folder}/output.txt"
    if os.path.exists(folder):
        shutil.rmtree(folder)

    try:
        os.mkdir(folder)
    except FileExistsError:
        pass

    cases = 100
    agents = [
        # RandomAgent(rng=random.Random(seed)),
        # SimpleRandomReinforcementAgent(rng=random.Random(seed)),
        # SimpleReinforcementAgent(rng=random.Random(seed)),
        # SimpleAgentNoEpsilon(rng=random.Random(seed)),
        # SimpleAgentMidEpsilon(rng=random.Random(seed)),
        # SimpleAgentHighEpsilon(rng=random.Random(seed)),
        # SimpleAgentMidAlpha(rng=random.Random(seed)),
        # SimpleAgentHighAlpha(rng=random.Random(seed)),
        DynamicEpsilonAgent(
            rng=random.Random(seed),
            alpha=0.02,
        ),
        DynamicEpsilonAgent(
            rng=random.Random(seed),
            alpha=0.04,
        ),
        DynamicEpsilonAgent(
            rng=random.Random(seed),
            alpha=0.06,
        ),
        DynamicEpsilonAgent(
            rng=random.Random(seed),
            alpha=0.08,
        ),
        DynamicEpsilonAgent(
            rng=random.Random(seed),
            alpha=0.1,
        ),
        DynamicEpsilonAgent(
            rng=random.Random(seed),
            alpha=0.12,
        ),
        DynamicEpsilonAgent(
            rng=random.Random(seed),
            alpha=0.14,
        ),
        DynamicEpsilonAgent(
            rng=random.Random(seed),
            alpha=0.16,
        ),
        DynamicEpsilonAgent(
            rng=random.Random(seed),
            alpha=0.18,
        ),
        DynamicEpsilonAgent(
            rng=random.Random(seed),
            alpha=0.20,
        ),
        DynamicEpsilonAgent(
            rng=random.Random(seed),
            alpha=0.22,
        ),
        DynamicEpsilonAgent(
            rng=random.Random(seed),
            alpha=0.24,
        ),
        DynamicEpsilonAgent(
            rng=random.Random(seed),
            alpha=0.26,
        ),
        DynamicEpsilonAgent(
            rng=random.Random(seed),
            alpha=0.28,
        ),
    ]
    grid = EarlyExitGrid(build_board_array([], size=size))

    # for agent in agents:
    #     agent.rng.seed(seed)
    #     runner = Runner(
    #         env=grid,
    #         agent=agent,
    #     )

    #     print(f"Running {agent.__class__.__name__}")
    #     results = runner.run_episodes(num_episodes=cases, render=True)
    #     runner.render_analytics(results)
    #     runner.build_heatmap(
    #         results, folder=folder, agent_name=agent.__class__.__name__, cases=cases
    #     )
    #     print("")

    # compare across cases
    all_results: dict[str, list[RunnerReturn]] = {str(agent): [] for agent in agents}
    grids = [EarlyExitGrid(build_board_array([], size=size)) for _ in range(cases)]
    for agent in agents:
        for grid in grids:
            agent.rng.seed(seed)
            runner = Runner(env=grid, agent=agent)
            results = runner.run_episodes(num_episodes=cases, render=False)
            all_results[str(agent)].extend(results)
        runner = Runner(
            env=grid,
            agent=agent,
        )
        print(f"Running {str(agent)}")
        runner.render_analytics(all_results[str(agent)])
        print("")
