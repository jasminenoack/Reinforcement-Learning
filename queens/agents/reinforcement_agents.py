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
from queens.dtos import Observation, Result, RunnerReturn


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

    def observe_step(self, result: Result):
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
