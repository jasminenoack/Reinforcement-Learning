from dataclasses import dataclass
from tic_tac_logic.sample_grids import (
    get_testing_sample_grids,
)

from tic_tac_logic.constants import Result
from tic_tac_logic.agents.mask_agent import MaskAgent
from tic_tac_logic.env.grid import Grid
from tic_tac_logic.constants import O
from tic_tac_logic.runner import print_grid
import logging
from tic_tac_logic.agents.masks import generate_pool_masks

logging.basicConfig(
    filename="tic_tac_logic/mask_agent.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


grids = get_testing_sample_grids()
envs = [Grid(g.grid) for g in grids]
# this is not ideal... we never use this again it's just to check the size
# so we can swap to others of the same size

rows = len(grids[0].grid)
columns = len(grids[0].grid[0])

# all_masks = generate_pool_masks(rows=rows, columns=columns)
# print(len(all_single_block_masks), "single block masks generated")

# for mask in generate_pool_masks(3, 3):
#     print(mask)

agent = MaskAgent(rows, columns, masks=generate_pool_masks(3, 3))
rounds_of_attempts = 100


@dataclass
class Counts:
    failure_count: int
    success_count: int

    def failure_probability(self) -> float:
        if self.failure_count + self.success_count == 0:
            return 0.0
        return self.failure_count / (self.failure_count + self.success_count) * 100


@dataclass
class MaskInfo:
    mask_type: str
    pattern: str
    symbol: str
    counts: Counts


def run_episode(
    agent: MaskAgent, grid: Grid, train: bool = False, explain: bool = False
) -> Result:
    agent.explain = explain
    grid.reset()
    agent.reset()
    if explain:
        logger.info("Starting a new episode with the following grid:")
        print_grid(grid)
    while not grid.lost()[0] and not grid.won()[0]:
        action = agent.act(grid.get_observation())
        step_result = grid.act(*action)
        if train:
            agent.learn(step_result)
        if explain:
            print_grid(grid)
    result = Result(
        actions=grid.actions,
        score=grid.score,
        won=grid.won()[0],
        q_table=agent.q_table if hasattr(agent, "q_table") else None,
        error=None,
    )
    return result


def mask_builder_view(agent: MaskAgent, envs: list[Grid]) -> None:
    results: list[Result] = []
    for i in range(rounds_of_attempts):
        print(f"Training episode {i + 1}/{rounds_of_attempts}")
        for env in envs:
            results.append(run_episode(agent, env, train=True))

    q_table = agent.q_table["masks"]

    x_placements: list[MaskInfo] = []
    o_placements: list[MaskInfo] = []
    for mask in q_table:
        pattern = mask.pattern
        symbol = mask.symbol_to_place
        if symbol == O:
            o_placements.append(
                MaskInfo(
                    mask_type=mask.name,
                    pattern=pattern,
                    symbol=symbol,
                    counts=Counts(
                        failure_count=q_table[mask].failure_count,
                        success_count=q_table[mask].success_count,
                    ),
                )
            )
        else:
            x_placements.append(
                MaskInfo(
                    mask_type=mask.name,
                    pattern=pattern,
                    symbol=symbol,
                    counts=Counts(
                        failure_count=q_table[mask].failure_count,
                        success_count=q_table[mask].success_count,
                    ),
                )
            )

    print("Q Table")
    print("")
    print("    Top X Placements failure rates")
    for placement in sorted(
        x_placements,
        key=lambda x: (-x.counts.failure_probability(), -x.counts.failure_count),
    )[:10]:
        formatted_pattern = placement.pattern.replace("\n", "|")
        print(
            f"        {formatted_pattern}  {placement.counts.failure_probability():.0f} ({placement.counts.failure_count}/{placement.counts.success_count})"
        )

    print("")
    print("    Top O Placements failure rates")
    for placement in sorted(
        o_placements,
        key=lambda x: (-x.counts.failure_probability(), -x.counts.failure_count),
    )[:10]:
        formatted_pattern = placement.pattern.replace("\n", "|")
        print(
            f"        {formatted_pattern}  {placement.counts.failure_probability():.0f} ({placement.counts.failure_count}/{placement.counts.success_count})"
        )


def print_grid(grid: Grid) -> None:
    original_grid = grid.initial_grid
    current_grid = grid.grid
    for original_row, current_row in zip(original_grid, current_grid):
        row: list[str] = []
        for original_cell, current_cell in zip(original_row, current_row):
            if original_cell == current_cell:
                row.append(f"{current_cell} ")
            else:
                row.append(f"{current_cell}*")
        logger.info("| ".join(row))
    logger.info("")


def render_analytics(results: list[Result], agent: MaskAgent, env: Grid) -> None:
    total_actions = sum(result.actions for result in results)
    total_score = sum(result.score for result in results)
    total_wins = sum(1 for result in results if result.won)
    errors = len([result for result in results if result.error])

    print(f"Average Actions per Episode: {total_actions / len(results):.2f}")
    print(f"Average Score per Episode: {total_score / len(results):.2f}")
    print(f"Error Rate: {errors / len(results) * 100:.2f}%")
    print(f"Win Rate: {total_wins / len(results) * 100:.2f}%")
    print_grid(env)
    print("-" * 40)


results: dict[int, list[Result]] = {}
mask_builder_view(agent, envs)
print("Done with training")
agent.fully_trained()
for env_number, env in enumerate(envs):
    results[env_number] = []
    for i in range(rounds_of_attempts):
        env.reset()
        results[env_number].append(run_episode(agent, env, train=False, explain=i == 0))
        logger.info("")
        logger.info("")
        logger.info("-" * 40)
        logger.info("")
        logger.info("")
    print(f"Results for env {env_number}:")
    render_analytics(results[env_number], agent, envs[env_number])
    print("")
