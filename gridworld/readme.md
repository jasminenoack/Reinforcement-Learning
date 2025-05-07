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

# Agent Types: 


## 🤖 Planned Agent Types

We aim to test different agent strategies and compare their performance across various environment configurations.

1. **Random Agent**  
   - Chooses actions at random with no learning or memory.  
   - Acts as a baseline for comparison.

2. **Trained Agent (Default Penalties)**  
   - Uses reinforcement learning (e.g., Q-learning).  
   - Learns over multiple episodes using a standard rewar structure.

3. **Trained Agent (Custom Penalties)**  
   - Same algorithm as (2), but trained in an environment with different penalty values (e.g., higher wall penalties).  
   - Used to explore how environment shaping affects learning.

4. **Designed Agent**  
   - A hardcoded or heuristic-based agent (e.g., always move right until blocked, then move down).  
   - Useful for comparing against learned behavior without any training time.

5. **Training Variants**  
   - Investigate how different training parameters affect agent performance:
     - Number of training episodes
     - Maximum steps per episode
     - Exploration rate (epsilon)
     - Learning rate and discount factor
     - what if the world was round
     - Large worlds
