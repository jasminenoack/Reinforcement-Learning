import os
import shutil
import random
from time import sleep
from typing import Any
from queens.agents.generic_agent import Agent
from queens.agents.random_agent import (
    RandomAgent,  # type: ignore
    RandomAgentAlsoByColumn,  # type: ignore
    RandomAgentByRow,  # type: ignore
)
from queens.agents.reinforcement_agents import (
    DynamicEpsilonAgent,  # type: ignore
    SimpleAgentHighAlpha,  # type: ignore
    SimpleAgentHighEpsilon,  # type: ignore
    SimpleAgentMidAlpha,  # type: ignore
    SimpleAgentMidEpsilon,  # type: ignore
    SimpleAgentNoEpsilon,  # type: ignore
    SimpleRandomReinforcementAgent,  # type: ignore
    DecayFailingPaths,  # type: ignore
    SimpleReinforcementAgent,  # type: ignore
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
        self, *, render: bool = False, render_result: bool = False, train: bool = True
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
            if train:
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
        if train:
            self.agent.observe_result(result)

        return result

    def run_episodes(
        self,
        num_episodes: int,
        *,
        render: bool = False,
        train: bool = True,
    ) -> list[RunnerReturn]:
        results: list[RunnerReturn] = []
        for _ in range(num_episodes):
            result = self.run_episode(render=False, render_result=render, train=train)
            results.append(result)
        return results

    def render_analytics(self, results: list[RunnerReturn]):
        solved = sum(1 for result in results if result.solved)
        failed = sum(1 for result in results if not result.solved)
        avg_moves = sum(result.moves for result in results) / len(results)
        max_moves = max(result.moves for result in results)
        min_moves = min(result.moves for result in results)
        print(f"Total solved: {solved}")
        print(f"Total failed: {failed}")
        print(f"Average moves: {avg_moves:.2f}")
        print(f"Max moves: {max_moves}")
        print(f"Min moves: {min_moves}")

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
    size = 8
    folder = f"output/queens"
    output_file = f"{folder}/output.txt"
    if os.path.exists(folder):
        shutil.rmtree(folder)

    try:
        os.mkdir(folder)
    except FileExistsError:
        pass

    training_cases = [
        10,
        100,
        1000,
        5_000,
        10_000,
        20_000,
    ]
    cases = 100
    agent_classes = [
        # RandomAgent,
        # RandomAgentByRow,
        # RandomAgentAlsoByColumn,
        # SimpleRandomReinforcementAgent,
        DecayFailingPaths,
        # SimpleReinforcementAgent(),
        # SimpleAgentNoEpsilon(),
        # SimpleAgentMidEpsilon(),
        # SimpleAgentHighEpsilon(),
        # SimpleAgentMidAlpha(),
        # SimpleAgentHighAlpha(),
        # DynamicEpsilonAgent(
        #
        #     alpha=0.02,
        # ),
        # DynamicEpsilonAgent(
        #
        #     alpha=0.04,
        # ),
        # DynamicEpsilonAgent(
        #
        #     alpha=0.06,
        # ),
        # DynamicEpsilonAgent(
        #
        #     alpha=0.08,
        # ),
        # DynamicEpsilonAgent(
        #
        #     alpha=0.1,
        # ),
        # DynamicEpsilonAgent(
        #
        #     alpha=0.12,
        # ),
        # DynamicEpsilonAgent(
        #
        #     alpha=0.14,
        # ),
        # DynamicEpsilonAgent(
        #
        #     alpha=0.16,
        # ),
        # DynamicEpsilonAgent(
        #
        #     alpha=0.18,
        # ),
        # DynamicEpsilonAgent(
        #
        #     alpha=0.20,
        # ),
        # DynamicEpsilonAgent(
        #
        #     alpha=0.22,
        # ),
        # DynamicEpsilonAgent(
        #
        #     alpha=0.24,
        # ),
        # DynamicEpsilonAgent(
        #
        #     alpha=0.26,
        # ),
        # DynamicEpsilonAgent(
        #
        #     alpha=0.28,
        # ),
    ]

    for agent_class in agent_classes:
        for training_case_count in training_cases:
            agent = agent_class(rng=random.Random(1))
            print(f"Training {agent} {training_case_count} cases")
            grid = EarlyExitGrid(build_board_array([], size=size))
            runner = Runner(env=grid, agent=agent)
            _ = runner.run_episodes(num_episodes=training_case_count, render=False)
            print(f"Finished training {agent} {training_case_count} cases")
            print(f"Running {agent} {cases} cases")
            grid = EarlyExitGrid(build_board_array([], size=size))
            results = runner.run_episodes(num_episodes=cases, render=False, train=False)
            runner.render_analytics(results)
            print("")

    #         # training agents:
    #         for agent in agents:
    #             print(f"Training {str(agent)}")
    #             runner = Runner(env=grid, agent=agent)
    #             results = runner.run_episodes(
    #                 num_episodes=training_case_count, render=False
    #             )

    # grid = EarlyExitGrid(build_board_array([], size=size))
    # # training agents:
    # for agent in agents:
    #     print(f"Training {str(agent)}")
    #     runner = Runner(env=grid, agent=agent)
    #     results = runner.run_episodes(num_episodes=cases, render=False)

    # # compare across cases
    # all_results: dict[str, list[RunnerReturn]] = {str(agent): [] for agent in agents}
    # grids = [EarlyExitGrid(build_board_array([], size=size)) for _ in range(cases)]
    # for agent in agents:
    #     for grid in grids:
    #         runner = Runner(env=grid, agent=agent)
    #         results = runner.run_episodes(num_episodes=cases, render=False)
    #         all_results[str(agent)].extend(results)
    #     runner = Runner(
    #         env=grid,
    #         agent=agent,
    #     )
    #     print(f"Running {str(agent)}")
    #     runner.render_analytics(all_results[str(agent)])
    #     print("")
