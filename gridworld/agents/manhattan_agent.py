from collections import defaultdict
import random
from gridworld.agents.generic_agent import Agent
from gridworld.utils import (
    DOWN,
    REVERSED_ACTIONS,
    RIGHT,
    SIMPLE_ACTIONS,
    UP,
    LEFT,
    Step,
)


class ManhattanAgent(Agent):

    def reset(self, **kwargs):
        self._action_cache = defaultdict(set)
        self._backtrack = []

    def act(self, state):
        row = state[0]
        col = state[1]
        row_goal = self.goal[0]
        col_goal = self.goal[1]

        best_options = []

        if row < row_goal:
            best_options.append(DOWN)
        if row > row_goal:
            best_options.append(UP)
        if col < col_goal:
            best_options.append(RIGHT)
        if col > col_goal:
            best_options.append(LEFT)

        previously_attempted = self._action_cache[state]
        best_options = list(set(best_options) - previously_attempted)
        any_options = list(set(SIMPLE_ACTIONS) - previously_attempted)

        action = None
        if best_options:
            action = random.choice(best_options)
            self._action_cache[state].add(action)

        elif any_options:
            action = random.choice(any_options)
            self._action_cache[state].add(action)
            self._backtrack.append(REVERSED_ACTIONS[action])
        elif self._backtrack:
            action = self._backtrack.pop()

        if action:
            return action

        # if something is wrong
        raise ValueError(
            f"Agent is at {state} it believes this is the goal: {self.goal}. ",
            "Please check the agent's config.",
        )

    def observe(self, step: Step):
        self._action_cache[step.start].add(step.action)

        # if we aren't trying to go backwards
        reversed_action = REVERSED_ACTIONS[step.action]
        if reversed_action not in self._action_cache[step.new_state]:
            self._backtrack.append(REVERSED_ACTIONS[step.action])
