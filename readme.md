This is a code base that is meant to allow practice with reinforcement learning (RL) in a simple environment. The goal is to create a series of RL agents that can learn to interact with various games and environments. The code is designed to be modular and extensible, allowing for easy addition of new environments and agents.

All of the games are placement games and not interaction games. The goal is to correctly place an object once and not move it again. Like in chess, once you remove your hand that's a wrap. The goal is to learn the best (or any effective) placement strategy for each game.

Within each game, there will likely be various stages. Solving a simple sudoku is quite different from solving a complex sudoku even if they are the same size. If you have a 4X4 starting with 12 numbers, it's likely trivial, if you have a 4X4 starting with 2 numbers, it's potentially impossible. The goal is to create a series of agents that can learn to play the game and solve the puzzles.

The initial goal is not for the same agent to solve all the games, but to create a series of agents that can learn to play the game and solve the puzzles. The goal is to create a series of agents that can learn to play the game and solve the puzzles. The goal is to create a series of agents that can learn to play the game and solve the puzzles.

# Games 

1. Gridworld (RL fundamentals)
2. Tic-Tac-Toe (strategy and rewards)
3. 8 Queens (search and constraints)
3. Tic-Tac-Logic (rule-driven grids)
4. Small Sudoku (structured logic with partial observability)
5. Large Sudoku (full challenge)

# Open questions 

1. Are we able to understand the rules that have been created post training the agent
2. Can we understand the steps taken by the agent to solve the puzzle
3. How are the agents doing? 
4. Can we start to consider the difference between a human and an agent
5. How does the amount of reinforcement change the agent (if we can tell it immediately it's wrong vs it needing to go several steps before it gets a reward or penalty)
6. How do we know when the agent is done learning?
7. Is the AI useful and in what way, given these are functionally solved algorithms


