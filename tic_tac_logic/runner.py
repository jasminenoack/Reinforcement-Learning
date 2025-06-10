# import math
from time import sleep
from typing import Any
from tic_tac_logic.env.grid import Grid
from tic_tac_logic.sample_grids import (
    # get_one_off_grid,
    get_easy_grid,
)

# from tic_tac_logic.agents.shaping_agents import RLShapingBasedAgent
# from tic_tac_logic.agents.failure_learning_agent import FailureAgent
from tic_tac_logic.agents.base_agent import Agent
from tic_tac_logic.constants import Result
from tic_tac_logic.agents.mask_agent import MaskAgent

# import numpy as np


def print_grid(grid: Grid) -> None:
    for row in grid.grid:
        print(" | ".join(row))
    print()


def run_episode(
    agent: Agent, grid: Grid, render: bool = False, train: bool = False
) -> Result:
    # This function would contain the logic to run an episode with the agent
    # For now, we will just print the initial grid and agent's actions
    grid.reset()
    if render:
        print("Running episode...")
        print_grid(grid)
    while not grid.lost()[0] and not grid.won()[0]:
        try:
            action = agent.act(grid.get_observation())
            step_result = grid.act(*action)
        except ValueError as e:
            return Result(
                actions=grid.actions,
                score=grid.score,
                won=False,
                q_table=agent.q_table if hasattr(agent, "q_table") else None,
                error=str(e),
            )
        if train:
            agent.learn(step_result)
        if render:
            print(f"Agent chose action: {action}")
            print(f"Scored: {step_result.score}")
            print(f"New q table: {agent.q_table.get(action[0], 'N/A')}")
            print_grid(grid)
            sleep(0.5)
    result = Result(
        actions=grid.actions,
        score=grid.score,
        won=grid.won()[0],
        q_table=agent.q_table if hasattr(agent, "q_table") else None,
        error=None,
    )
    # if render:
    #     print(grid.lost()[1])
    #     print(result)
    return result


def render_analytics(results: list[Result], agent: Agent, env: Grid) -> None:
    total_actions = sum(result.actions for result in results)
    total_score = sum(result.score for result in results)
    total_wins = sum(1 for result in results if result.won)
    errors = len([result for result in results if result.error])

    print(f"Average Actions per Episode: {total_actions / len(results):.2f}")
    print(f"Average Score per Episode: {total_score / len(results):.2f}")
    print(f"Length of failures: {len(agent.q_table['failures'])}")
    print(f"Error Rate: {errors / len(results) * 100:.2f}%")
    print(f"Win Rate: {total_wins / len(results) * 100:.2f}%")
    print_grid(env)
    print("-" * 40)


def run_episodes(
    agent: Agent,
    grid: Grid,
    episodes: int = 10,
    render: bool = False,
    train: bool = False,
) -> list[Result]:
    results: list[Result] = []
    for _ in range(episodes):
        grid.reset()
        result = run_episode(agent, grid, train=train)
        results.append(result)
    if render:
        render_analytics(results, agent, grid)
    # print_grid(grid)
    # print(grid.lost()[1])
    return results


if __name__ == "__main__":
    training_count = 2000
    training_rounds = 50
    non_training_count = 100
    grid = get_easy_grid()
    agent = MaskAgent(len(grid), len(grid[0]))
    grid = Grid(grid)
    for _ in range(100):
        grid.reset()
        # print_grid(grid)
        run_episode(agent, grid, train=True)
        # print_grid(grid)
        masks = agent.q_table["masks"]
        # sleep(0.3)
        # print("")
        # print("")
    print("Masks learned:")
    probably_failure: list[list[Any]] = []
    probably_success: list[list[Any]] = []
    unknown: list[list[Any]] = []
    # for mask, data in agent.q_table["masks"].items():
    #     name = mask.split("-")[1:]
    #     print("    ", name)
    #     successes = data["non_failure_count"]
    #     failures = data["failure_count"]
    #     total = successes + failures
    #     fail_probability = math.ceil(failures / total)
    #     if total < 5:
    #         unknown.append(mask)
    #     elif fail_probability > 0.9:
    #         probably_failure.append(mask)
    #     elif fail_probability < 0.3:
    #         probably_success.append(mask)
    #     else:
    #         unknown.append(mask)

    print("Probably failure masks:")
    for mask in sorted(probably_failure):
        print("    ", mask)
    print("")
    print("Probably success masks:")
    for mask in sorted(probably_success):
        print("    ", mask)
    print("")
    print("Unknown masks:")
    for mask in sorted(unknown):
        print("    ", mask)

    # print(f"Running {non_training_count} episodes before training...")
    # _ = run_episodes(agent, grid, episodes=non_training_count, render=True, train=False)
    # for _ in range(training_rounds):
    #     print(f"Running {training_count} Training episodes...")
    #     agent.reset()
    #     _ = run_episodes(agent, grid, episodes=training_count, render=True, train=True)
    # print(f"Running {non_training_count} episodes after training...")
    # _ = run_episodes(agent, grid, episodes=non_training_count, render=True, train=False)

    # agent = FailureAgent(grid.grid)
    # for _ in range(4):
    #     run_episode(grid=grid, agent=agent, render=True, train=True)

    #     print("Q Table failures:")
    #     failures = agent.q_table["failures"]
    #     for failure in failures:
    #         print(
    #             "    ",
    #             failure.location,
    #             "[",
    #             failure.applicable_area,
    #             "]",
    #             failure.symbol,
    #         )

    #     print("--" * 20)
