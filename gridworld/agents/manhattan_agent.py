import random
from typing import Any
from collections.abc import MutableSet

from gridworld.agents.generic_agent import Agent
from gridworld.utils import (
    DOWN,
    RIGHT,
    SIMPLE_ACTIONS,
    UP,
    LEFT,
    Step,
)


class ManhattanAgent(Agent):

    def reset(self, **kwargs: Any) -> None:
        self._tried: dict[tuple[int, int], MutableSet[str]] = {}

    def act(self, state: tuple[int, int]) -> str:
        row, col = state
        row_goal, col_goal = self.goal

        # Preferred directions (Manhattan-style)
        preferred = []
        if row < row_goal:
            preferred.append(DOWN)
        if row > row_goal:
            preferred.append(UP)
        if col < col_goal:
            preferred.append(RIGHT)
        if col > col_goal:
            preferred.append(LEFT)

        tried = self._tried.get(state, set())
        options = [a for a in preferred if a not in tried]

        # Fallback to any untried direction
        if not options:
            options = [a for a in SIMPLE_ACTIONS if a not in tried]

        if options:
            action = random.choice(options)
            self._tried.setdefault(state, set()).add(action)
            return action

        raise ValueError(f"No available moves from {state}; stuck or misconfigured.")

    def observe(self, step: Step) -> None:
        # Optional: reinforce that this action was tried (may be redundant)
        self._tried.setdefault(step.start, set()).add(step.action)
