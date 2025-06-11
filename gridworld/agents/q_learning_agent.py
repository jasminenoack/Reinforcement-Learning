from collections import defaultdict
from random import Random
from gridworld.agents.generic_agent import Agent
from gridworld.utils import SIMPLE_ACTIONS, Step


class QLearningAgent(Agent):
    def __init__(self, *, rng: Random | None = None, **kwargs):
        self.actions = SIMPLE_ACTIONS
        self.q_table: defaultdict[tuple[int, int], dict[str, float]] = defaultdict(
            lambda: {action: 0.0 for action in self.actions}
        )
        # exploration rate
        self.epsilon = 0.1
        # learning rate (how fast Q-values update)
        self.alpha = 0.1
        # discount factor (importance of future reward)
        self.gamma = 0.9
        self.rng = rng or Random()
        self.reset()

    def act(self, state):
        """
        using epsilon sometimes make a random choice

        otherwise choose the option that you know from history is one
        of the best options
        """
        if self.rng.random() < self.epsilon:
            return self.rng.choice(self.actions)

        q_values = self.q_table[state]
        max_q_value = max(q_values.values())

        best_actions = [
            action for action, q_value in q_values.items() if q_value == max_q_value
        ]
        return self.rng.choice(best_actions)

    def reset(self):
        pass

    def observe(self, step: Step):
        state = step.start
        next_state = step.new_state
        action = step.action
        reward = step.reward
        done = step.done

        current_q = self.q_table[state][action]

        if done:
            best_future_q = 0
        else:
            best_future_q = max(self.q_table[next_state].values())

        updated_q = current_q + self.alpha * (
            reward + self.gamma * best_future_q - current_q
        )
        self.q_table[state][action] = updated_q
