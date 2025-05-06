# 🧭 What is Gridworld?

**Gridworld** is a simple, grid-based environment often used to teach reinforcement learning. The world is a 2D grid (like 5×5), and an agent must learn how to move through the grid to reach a goal.

# Key Features:

* **Grid**: A matrix where each cell is either empty, a wall, the goal, or the agent.
* **Agent**: Can move up, down, left, or right (unless blocked by edges or walls).
* **State**: Usually represented by the agent’s position.
* **Actions**: The movement directions (4 total).
* **Rewards**:

  * +1 for reaching the goal.
  * -0.01 per step to encourage shorter paths.
  * Optionally -1 for hitting walls or bad tiles.

# Why Gridworld?

It’s ideal for learning:

* **Exploration vs. exploitation** (should the agent try something new or stick with what it knows?)
* **Delayed rewards**
* **Q-learning or policy learning**
* **How an agent improves over time with feedback**
