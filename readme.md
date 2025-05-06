# RL Puzzle Agents

This codebase is designed to support exploration and practice with reinforcement learning (RL) in structured, logic-based environments. The goal is to create modular, extensible agents that learn how to interact with grid-based puzzles through placement — not dynamic interaction.

These environments are all **placement games**: once an object is placed, it stays. Like placing a queen in chess or a number in Sudoku — no backsies. The challenge is to learn effective or optimal placement strategies from reward signals, not handcrafted rules.

Each game includes multiple difficulty stages. For example, a 4×4 Sudoku with 12 given numbers may be trivial, but one with only 2 may be unsolvable. Agents should learn to generalize across these variants.

The aim is not to build one universal agent, but rather to understand what it takes for an agent to solve each class of puzzle effectively.

## 🧩 Games

1. Gridworld (RL fundamentals)
2. Tic-Tac-Toe (strategy and rewards)
3. 8 Queens (search and constraints)
4. Tic-Tac-Logic (rule-driven grids)
5. Small Sudoku (structured logic with partial observability)
6. Large Sudoku (full challenge)

## ❓ Open Questions

1. Can we understand the rules the agent has learned, post-training?
2. Can we interpret the agent’s steps to understand how it solves puzzles?
3. How well are the agents doing — and by what metrics?
4. In what ways do human strategies differ from learned strategies?
5. How does reward timing (immediate vs delayed feedback) impact learning?
6. How do we know when an agent has "finished" learning?
7. Are these agents useful — or just a fun way to replicate solved problems?

