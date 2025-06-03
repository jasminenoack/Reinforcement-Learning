from time import sleep
from tic_tac_logic.env.grid import Grid
from tic_tac_logic.sample_grids import (
    # get_one_off_grid,
    get_easy_grid,
)
from tic_tac_logic.agents.simple_rl_agents import RLAgent
from tic_tac_logic.constants import Result
import numpy as np


def print_grid(grid: Grid) -> None:
    for row in grid.grid:
        print(" | ".join(row))
    print()


def run_episode(
    agent: RLAgent, grid: Grid, render: bool = False, train: bool = False
) -> Result:
    # This function would contain the logic to run an episode with the agent
    # For now, we will just print the initial grid and agent's actions
    if render:
        print("Running episode...")
        print_grid(grid)
    while not grid.lost()[0] and not grid.won()[0]:
        action = agent.act(grid.get_observation())
        step_result = grid.act(*action)
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
    )
    if render:
        print(grid.lost()[1])
        print(result)
    return result


def create_ascii_heatmap(
    raw_data: list[list[float]], characters: str = " `'.,-~:;=!*%#@$"
):
    """
    Creates an ASCII heat map from a 2D array.

    Args:
    data: 2D array of numeric data.
    characters: String of ASCII characters for mapping.

    Returns:
    String of the ASCII heat map.
    """
    data = np.array(raw_data)
    print(data)
    normalized_data = (data - np.min(data)) / (
        np.max(data) - (np.min(data)) + -0.000000000000001
    )
    num_chars = len(characters)
    indices = (normalized_data * (num_chars - 1)).astype(int)
    result: list[str] = []
    for row in indices:
        row_indices = [int(i) for i in row]
        result.append(" ".join(characters[index] for index in row_indices))
    print("\n".join(result))


def render_analytics(results: list[Result]) -> None:
    total_actions = sum(result.actions for result in results)
    total_score = sum(result.score for result in results)
    total_wins = sum(1 for result in results if result.won)

    print(f"Average Actions per Episode: {total_actions / len(results):.2f}")
    print(f"Average Score per Episode: {total_score / len(results):.2f}")
    # print(f"Total Wins: {total_wins} out of {len(results)}")
    print(f"Win Rate: {total_wins / len(results) * 100:.2f}%")
    print("-" * 40)
    raw_max_scores: list[list[float]] = []
    raw_min_scores: list[list[float]] = []
    coordinates: list[tuple[int, int]] = list(
        results[0].q_table.keys() if results[0].q_table else []
    )
    max_row = max([cord[0] for cord in coordinates])
    max_column = max([cord[1] for cord in coordinates])
    if max_row and max_column:
        for row in range(max_row + 1):
            row_maxes: list[float] = []
            row_mins: list[float] = []
            for column in range(max_column + 1):
                result = results[-1]
                if result.q_table:
                    values = result.q_table[(row, column)].values()
                    row_maxes.append(max(values))
                    row_mins.append(min(values))
            raw_max_scores.append(row_maxes)
            raw_min_scores.append(row_mins)

    print("Max Scores Heatmap:")
    create_ascii_heatmap(raw_max_scores)
    print("-" * 40)
    # print("Min Scores Heatmap:")
    # create_ascii_heatmap(raw_min_scores)
    # print("-" * 40)


def run_episodes(
    agent: RLAgent,
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
        render_analytics(results)
    # print_grid(grid)
    # print(grid.lost()[1])
    return results


if __name__ == "__main__":
    training_count = 1000
    training_rounds = 100
    non_training_count = 100
    grid = get_easy_grid()
    agent = RLAgent(grid)
    grid = Grid(grid)
    # for _ in range(3):
    #     grid.reset()
    #     run_episode(agent, grid, render=True, train=True)
    #     sleep(5)
    #     print("")
    #     print("")
    print(f"Running {non_training_count} episodes before training...")
    _ = run_episodes(agent, grid, episodes=non_training_count, render=True, train=False)
    for _ in range(training_rounds):
        print(f"Running {training_count} Training episodes...")
        agent.reset()
        _ = run_episodes(agent, grid, episodes=training_count, render=True, train=True)
    print(f"Running {non_training_count} episodes after training...")
    _ = run_episodes(agent, grid, episodes=non_training_count, render=True, train=False)
