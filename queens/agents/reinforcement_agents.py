"""
That’s an excellent long-term direction — shifting from “reward-only memory” to **abstract learning of structure** is where you move from imitation of trial-and-error toward something closer to intelligence.

Let’s lay out a progression from what you have → what you could build to let the agent **internalize structure** like a human might.

---

## ✅ Current Agent (What You Have)

**Strategy**:

* Remember best full or partial paths
* Reuse them, slightly mutated
* Score based on outcome + conflict-free length

**Learning Signal**:

* Sparse reward (only success or conflict)

**What It Learns**:

* Nothing about *why* moves fail
* No general rules

---

## 🔁 Stage 1: Track *why* moves fail

**Add**: Annotate failures — row/column/diagonal — and **track frequency of each failure type**.

Now you can learn:

* “My queens keep bumping diagonally.”
* “Columns aren't the main problem.”

That builds **error awareness**, which is foundational to human learning.

---

## 🧱 Stage 2: Learn feature-based constraints

Start tracking basic **features** of the board state or action like:

* Distance from other queens
* Number of diagonals threatened
* Is the column already used?

Then:

* Penalize high-threat moves
* Favor low-conflict options

This is like teaching the agent, "Bad placements have patterns."

---

## 🧠 Stage 3: Internal model of threats

Have the agent build a **threat map**:

* A board of counts: "how many queens threaten this square"
* Choose moves that minimize threat

Now it behaves more like a human: “Place a queen where it won’t be attacked.”

This could be rule-based, or learned from past successful boards.

---

## 🔄 Stage 4: Learn generalizable patterns

Instead of memorizing paths, memorize:

* "When I place on (row, col), what tends to go wrong?"
* Build probability maps of failure per position
* Over time, learn priors like “corners are good” or “center early is bad”

This is **concept learning**, not just sequence memorization.

---

## 🤖 Stage 5: Predictive modeling (hard mode)

Build a model that **predicts whether a board state is solvable**.

Use a classifier (even a basic one) trained on:

* Partial board state → solvable yes/no

Now your agent can ask:

> “Does this placement likely lead somewhere good?”

---

## 🧬 Optional: Evolutionary strategies

Instead of acting step-by-step, evolve full 8-queen boards:

* Mutate
* Score
* Keep the best

This drops planning entirely in favor of *population-level learning* — works well when interpretability isn’t important.

---

### TL;DR — if a human were learning this:

1. They’d fail, then notice *why* (diagonals!)
2. They’d start trying to avoid repeated mistakes
3. They’d develop an internal “threat map” and act to avoid conflict
4. They’d form reusable patterns and rules, not full board memories

You can give the agent the same growth path — and it'll start to look a lot more like thinking.

Let me know which step sounds most interesting and we can prototype it.
"""

from collections import defaultdict
import random
from typing import Any
from queens.agents.random_agent import RandomAgent
from queens.dtos import Observation, StepResult, RunnerReturn
from numpy.typing import NDArray
import numpy as np


class SimpleRandomReinforcementAgent(RandomAgent):
    epsilon = 0.1

    def __init__(self, *, rng: random.Random | None = None, **kwargs: Any):
        super().__init__(rng=rng)
        self.current: list[tuple[int, int]] = []
        self.best_options: defaultdict[int, list[list[tuple[int, int]]]] = defaultdict(
            list
        )

    def act(self, observation: Observation) -> Any:
        random_num = self.rng.random()
        if len(self.following_path) > len(self.current) and random_num > self.epsilon:
            move = self.following_path[len(self.current)]
        else:
            move = super().act(observation=observation)

        self.current.append(move)
        return move

    def observe_step(self, result: StepResult):
        pass

    def _best_score(self) -> int:
        if not self.best_options:
            return -10000000000
        return max(self.best_options.keys())

    def _sample_best_paths(self) -> list[tuple[int, int]]:
        if not self.best_options:
            return []

        scores = sorted(self.best_options.keys(), reverse=True)
        if len(scores) == 1:
            best_path = self.rng.choice(self.best_options[self._best_score()])
        else:
            best_path = self.rng.choice(
                self.best_options[scores[0]] + self.best_options[scores[1]]
            )
        return best_path

    def reset(self, **kwargs: Any):
        self.following_path = self._sample_best_paths()
        self.current = []

    def observe_result(self, result: RunnerReturn):
        seen_row: set[int] = set()
        seen_column: set[int] = set()
        seen_diagonal: set[int] = set()
        seen_reverse_diagonal: set[int] = set()

        path: list[tuple[int, int]] = []

        for item in result.trajectory:
            row, col = item.action

            new_row = row not in seen_row
            new_col = col not in seen_column
            new_diag = (row - col) not in seen_diagonal
            new_reverse_diag = (row + col) not in seen_reverse_diagonal

            if not new_row or not new_col or not new_diag or not new_reverse_diag:
                break
            seen_row.add(row)
            seen_column.add(col)
            seen_diagonal.add(row - col)
            seen_reverse_diagonal.add(row + col)

            path.append((row, col))

        modified_score = result.score + len(path)
        if (
            modified_score >= self._best_score()
            and path not in self.best_options[modified_score]
        ):
            best = path.copy()
            self.best_options[modified_score].append(best)


# class ThoughtfulReinforcementAgent(SimpleRandomReinforcementAgent):
#     """
#     This is an interesting idea, but it's not really working in a learning way
#     it's just an overly complex heuristic forumlization.
#     """

#     def __init__(self, *args: Any, **kwargs: Any):
#         super().__init__(*args, **kwargs)
#         self.q_table = {
#             0: {
#                 FailureType.ROW: 0,
#                 FailureType.COLUMN: 0,
#                 FailureType.DIAGONAL: 0,
#                 FailureType.REVERSE_DIAGONAL: 0,
#             },
#             1: {
#                 FailureType.ROW: 0,
#                 FailureType.COLUMN: 0,
#                 FailureType.DIAGONAL: 0,
#                 FailureType.REVERSE_DIAGONAL: 0,
#             },
#         }

#     def _select_row(self, board: NDArray[np.int_]) -> int:
#         by_score: dict[int, int] = {}
#         for i in range(8):
#             row_sum = np.sum(board[i])
#             by_score[i] = self.q_table[int(row_sum)][FailureType.ROW]
#         highest_score = max(by_score.values())
#         best_rows = [k for k, v in by_score.items() if v == highest_score]
#         return self.rng.choice(best_rows)

#     def _select_column(self, board: NDArray[np.int_]) -> int:
#         by_score: dict[int, int] = {}
#         for i in range(8):
#             column_sum = np.sum(board[:, i])
#             by_score[i] = self.q_table[int(column_sum)][FailureType.COLUMN]
#         highest_score = max(by_score.values())
#         best_columns = [k for k, v in by_score.items() if v == highest_score]
#         return self.rng.choice(best_columns)

#     def observe_step(self, result: StepResult):
#         # no epsilon here.
#         reward = result.reward
#         failure_type = result.failure_type
#         if failure_type == FailureType.ROW:
#             row_sum = np.sum(result.board_state.board[result.action[0]]) - 1
#             self.q_table[row_sum][failure_type] += reward
#         elif failure_type == FailureType.COLUMN:
#             column_sum = np.sum(result.board_state.board[:, result.action[1]]) - 1
#             self.q_table[int(column_sum)][failure_type] += reward


class SimpleReinforcementAgent(RandomAgent):
    """
    This is a simple reinforcement agent that uses a Q-table to learn from its actions.
    It uses a simple epsilon-greedy strategy to explore the action space.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.q_table: dict[tuple[int, int], float] = defaultdict(float)
        self.alpha = 0.1  # Learning rate
        self.epsilon = 0.1

    def _random_unseen_location(self, observation: Observation) -> tuple[int, int]:
        options = [
            (r, c)
            for r in range(observation.board_state.board.shape[0])
            for c in range(observation.board_state.board.shape[0])
            if (r, c) not in self.seen
        ]
        return self.rng.choice(options)

    def _select_location(self, board: NDArray[np.int_]) -> tuple[int, int]:
        by_score: dict[tuple[int, int], float] = {}
        for row in range(board.shape[0]):
            for column in range(board.shape[0]):
                if (row, column) in self.seen:
                    continue
                by_score[(row, column)] = self.q_table[(row, column)]
        highest_score = max(by_score.values())
        best_locations = [k for k, v in by_score.items() if v == highest_score]
        return self.rng.choice(best_locations)

    def act(self, observation: Observation) -> tuple[int, int]:
        if self.rng.random() < self.epsilon:
            location = self._random_unseen_location(observation)
        else:
            location = self._select_location(observation.board_state.board)
        self.seen.add(location)
        return location

    def observe_step(self, result: StepResult):
        # Update Q-value based on the action taken and the reward received
        state = result.action
        reward = result.reward
        self.q_table[state] += self.alpha * (reward - self.q_table[state])

    def reset(self, **kwargs: Any):
        self.seen: set[tuple[int, int]] = set()


class SimpleAgentMidEpsilon(SimpleReinforcementAgent):
    """
    This is a simple reinforcement agent that uses a Q-table to learn from its actions.
    It uses a simple epsilon-greedy strategy to explore the action space.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.epsilon = 0.5


class SimpleAgentNoEpsilon(SimpleReinforcementAgent):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.epsilon = 0.0


class SimpleAgentHighEpsilon(SimpleReinforcementAgent):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.epsilon = 0.9


class SimpleAgentMidAlpha(SimpleReinforcementAgent):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.alpha = 0.5


class SimpleAgentHighAlpha(SimpleReinforcementAgent):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.alpha = 0.9
