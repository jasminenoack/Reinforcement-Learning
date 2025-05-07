import random
from gridworld.agents.generic_agent import Agent
from gridworld.utils import DOWN, RIGHT, UP, LEFT


class ManhattanAgent(Agent):

    def reset(self, **kwargs): ...

    def act(self, state):
        row = state[0]
        col = state[1]
        row_goal = self.goal[0]
        col_goal = self.goal[1]

        valid_options = []

        if row < row_goal:
            valid_options.append(DOWN)
        if row > row_goal:
            valid_options.append(UP)
        if col < col_goal:
            valid_options.append(RIGHT)
        if col > col_goal:
            valid_options.append(LEFT)

        if valid_options:
            return random.choice(valid_options)

        # if something is wrong
        raise ValueError(
            f"Agent is at {state} it believes this is the goal: {self.goal}. ",
            "Please check the agent's config.",
        )

    def observe(self, *args, **kwargs):
        pass
