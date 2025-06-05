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
agent = MaskAgent(grid=grids[0].grid)
rounds_of_attempts = 10


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
    for _ in range(rounds_of_attempts):
        for env in envs:
            results.append(run_episode(agent, env, train=True))

    q_table = agent.q_table["masks"]

    x_placements: list[MaskInfo] = []
    o_placements: list[MaskInfo] = []
    for mask in q_table:
        mask_type = mask.mask_type
        pattern = mask.pattern
        symbol = mask.symbol
        if symbol == O:
            o_placements.append(
                MaskInfo(
                    mask_type=mask_type.__name__,
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
                    mask_type=mask_type.__name__,
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
    print("    X Placements failure rates")
    for placement in sorted(x_placements, key=lambda x: x.pattern):
        print(
            f"        {placement.pattern}  {placement.counts.failure_probability():.0f} ({placement.counts.failure_count}/{placement.counts.success_count})"
        )

    print("")
    print("    O Placements failure rates")
    for placement in sorted(o_placements, key=lambda x: x.pattern):
        print(
            f"        {placement.pattern}  {placement.counts.failure_probability():.0f} ({placement.counts.failure_count}/{placement.counts.success_count})"
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


mask_builder_view(agent, envs)
print("Done with training")
for env in envs:
    env.reset()
    run_episode(agent, env, train=False, explain=True)
    logger.info("")
    logger.info("")
    logger.info("-" * 40)
    logger.info("")
    logger.info("")
