from collections import defaultdict
import random
from typing import Any

from gridworld.agents.generic_agent import Agent
from gridworld.utils import SIMPLE_ACTIONS, Step


class LostAgent(Agent):
    def __init__(self, *, rng: random.Random | None = None, **kwargs: Any) -> None:
        """
        This is rather like the random agent, but it strongly prefers to not repeat a move
        """
        self.rng = rng or random.Random()
        self.reset()

    def act(self, state: tuple[int, int]) -> str:
        actions = self.previous_attempts[state]
        min_value = min(actions.values())
        best_actions = [
            action for action, value in actions.items() if value == min_value
        ]

        if best_actions:
            choice = self.rng.choice(best_actions)
            return choice
        return self.rng.choice(SIMPLE_ACTIONS)

    def observe(self, step: Step) -> None:
        start = step.start
        action = step.action
        self.previous_attempts[start][action] += 1

        reversed_step = step.get_reverse_action()
        if reversed_step:
            self.previous_attempts[reversed_step.start][reversed_step.action] += 1
            # Increment the count for the reverse action as well

    def reset(self, **kwargs: Any) -> None:
        self.previous_attempts: defaultdict[tuple[int, int], dict[str, int]] = defaultdict(
            lambda: {action: 0 for action in SIMPLE_ACTIONS}
        )


if __name__ == "__main__":
    assert SIMPLE_ACTIONS, "No actions defined!"
    agent = LostAgent()
    for i in range(10):
        action = agent.act((0, 0))
        print(f"Action {i}: {action}")
