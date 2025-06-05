from dataclasses import dataclass
from tic_tac_logic.sample_grids import (
    get_testing_sample_grids,
)

from tic_tac_logic.constants import Result
from tic_tac_logic.agents.mask_agent import MaskAgent
from tic_tac_logic.env.grid import Grid
from tic_tac_logic.constants import O

grids = get_testing_sample_grids()
envs = [Grid(g.grid) for g in grids]
# this is not ideal... we never use this again it's just to check the size
# so we can swap to others of the same size
agent = MaskAgent(grid=grids[0].grid)
rounds_of_attempts = 10


def run_episode(agent: MaskAgent, grid: Grid) -> Result:
    grid.reset()
    agent.reset()
    while not grid.lost()[0] and not grid.won()[0]:
        action = agent.act(grid.get_observation())
        step_result = grid.act(*action)
        agent.learn(step_result)
    result = Result(
        actions=grid.actions,
        score=grid.score,
        won=grid.won()[0],
        q_table=agent.q_table if hasattr(agent, "q_table") else None,
        error=None,
    )
    return result


results: list[Result] = []
for _ in range(rounds_of_attempts):
    for env in envs:
        results.append(run_episode(agent, env))

q_table = agent.q_table["masks"]


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


x_placements: list[MaskInfo] = []
o_placements: list[MaskInfo] = []
for mask in q_table:
    mask_type, pattern, symbol = mask.split("-")
    if symbol == O:
        o_placements.append(
            MaskInfo(
                mask_type=mask_type,
                pattern=pattern,
                symbol=symbol,
                counts=Counts(
                    failure_count=q_table[mask]["failure_count"],
                    success_count=q_table[mask]["non_failure_count"],
                ),
            )
        )
    else:
        x_placements.append(
            MaskInfo(
                mask_type=mask_type,
                pattern=pattern,
                symbol=symbol,
                counts=Counts(
                    failure_count=q_table[mask]["failure_count"],
                    success_count=q_table[mask]["non_failure_count"],
                ),
            )
        )


def probability_of_failure(MaskInfo: str) -> float: ...


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
