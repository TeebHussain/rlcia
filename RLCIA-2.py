import numpy as np
import random

GRID_SIZE = 100
OBSTACLE_RATIO = 0.2
START = (0, 0)
GOAL = (GRID_SIZE - 1, GRID_SIZE - 1)

# Initialize the grid with obstacles
grid = np.zeros((GRID_SIZE, GRID_SIZE))
for _ in range(int(OBSTACLE_RATIO * GRID_SIZE ** 2)):
    x, y = np.random.randint(0, GRID_SIZE, size=2)
    if (x, y) not in [START, GOAL]:
        grid[x, y] = -1  # -1 represents an obstacle

# Reward structure
REWARD_GOAL = 1
REWARD_STEP = 0
REWARD_OBSTACLE = -1

# Possible actions (Up, Down, Left, Right)
actions = {
    0: (-1, 0),  # Up
    1: (1, 0),   # Down
    2: (0, -1),  # Left
    3: (0, 1)    # Right
}

# Helper function to check valid moves
def is_valid_position(position):
    x, y = position
    return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE and grid[x, y] != -1
def initialize_q():
    """Initialize Q-table with zeros for each state-action pair."""
    return np.zeros((GRID_SIZE, GRID_SIZE, len(actions)))

def choose_action(state, q_table, epsilon):
    """Choose action based on epsilon-greedy policy."""
    if random.uniform(0, 1) < epsilon:
        return random.choice(list(actions.keys()))  # Explore
    else:
        x, y = state
        return np.argmax(q_table[x, y])  # Exploit

def get_reward(state):
    """Return reward based on state."""
    if state == GOAL:
        return REWARD_GOAL
    elif not is_valid_position(state):
        return REWARD_OBSTACLE
    else:
        return REWARD_STEP

def take_action(state, action):
    """Take action and return the next state."""
    x, y = state
    dx, dy = actions[action]
    next_state = (x + dx, y + dy)
    return next_state if is_valid_position(next_state) else state

def q_learning(episodes=5000, alpha=0.1, gamma=0.9, epsilon=0.1):
    """Q-learning algorithm implementation."""
    q_table = initialize_q()

    for episode in range(episodes):
        state = START
        done = False

        while not done:
            action = choose_action(state, q_table, epsilon)
            next_state = take_action(state, action)
            reward = get_reward(next_state)

            # Update Q-value for Q-learning
            x, y = state
            nx, ny = next_state
            best_next_action = np.argmax(q_table[nx, ny])
            q_table[x, y, action] += alpha * (reward + gamma * q_table[nx, ny, best_next_action] - q_table[x, y, action])

            state = next_state
            done = state == GOAL

    return q_table

def sarsa(episodes=5000, alpha=0.1, gamma=0.9, epsilon=0.1):
    """SARSA algorithm implementation."""
    q_table = initialize_q()

    for episode in range(episodes):
        state = START
        action = choose_action(state, q_table, epsilon)
        done = False

        while not done:
            next_state = take_action(state, action)
            reward = get_reward(next_state)
            next_action = choose_action(next_state, q_table, epsilon)

            # Update Q-value for SARSA
            x, y = state
            nx, ny = next_state
            q_table[x, y, action] += alpha * (reward + gamma * q_table[nx, ny, next_action] - q_table[x, y, action])

            state, action = next_state, next_action
            done = state == GOAL

    return q_table

# Parameters
EPISODES = 5000
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.1

# Run Q-learning and SARSA
q_table_qlearning = q_learning(episodes=EPISODES, alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON)
q_table_sarsa = sarsa(episodes=EPISODES, alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON)

# Comparison or Analysis (e.g., plotting success rates, analyzing learned Q-values)
