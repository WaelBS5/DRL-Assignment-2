# Remember to adjust your student ID in meta.xml
import numpy as np
import random
import gym
from gym import spaces
import copy
import math

# Define COLOR_MAP and TEXT_COLOR (needed for rendering, though not used in evaluation)
COLOR_MAP = {
    0: "#cdc1b4", 2: "#eee4da", 4: "#ede0c8", 8: "#f2b179", 16: "#f59563",
    32: "#f67c5f", 64: "#f65e3b", 128: "#edcf72", 256: "#edcc61", 512: "#edc850",
    1024: "#edc53f", 2048: "#edc22e", 4096: "#3c3a32", 8192: "#3c3a32"
}
TEXT_COLOR = {0: "black", 2: "black", 4: "black", 8: "white", 16: "white", 32: "white",
              64: "white", 128: "white", 256: "white", 512: "white", 1024: "white",
              2048: "white", 4096: "white", 8192: "white"}

# Game2048Env class (as provided)
class Game2048Env(gym.Env):
    def __init__(self):
        super(Game2048Env, self).__init__()
        self.size = 4
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.action_space = spaces.Discrete(4)
        self.actions = ["up", "down", "left", "right"]
        self.last_move_valid = True
        self.reset()

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.board

    def add_random_tile(self):
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            x, y = random.choice(empty_cells)
            self.board[x, y] = 2 if random.random() < 0.9 else 4

    def compress(self, row):
        new_row = row[row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        return new_row

    def merge(self, row):
        for i in range(len(row) - 1):
            if row[i] == row[i + 1] and row[i] != 0:
                row[i] *= 2
                row[i + 1] = 0
                self.score += row[i]
        return row

    def move_left(self):
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            new_row = self.compress(self.board[i])
            new_row = self.merge(new_row)
            new_row = self.compress(new_row)
            self.board[i] = new_row
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_right(self):
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            reversed_row = self.board[i][::-1]
            reversed_row = self.compress(reversed_row)
            reversed_row = self.merge(reversed_row)
            reversed_row = self.compress(reversed_row)
            self.board[i] = reversed_row[::-1]
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_up(self):
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            col = self.compress(self.board[:, j])
            col = self.merge(col)
            col = self.compress(col)
            self.board[:, j] = col
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def move_down(self):
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            reversed_col = self.board[:, j][::-1]
            reversed_col = self.compress(reversed_col)
            reversed_col = self.merge(reversed_col)
            reversed_col = self.compress(reversed_col)
            self.board[:, j] = reversed_col[::-1]
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def is_game_over(self):
        if np.any(self.board == 0):
            return False
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i, j] == self.board[i, j+1]:
                    return False
        for j in range(self.size):
            for i in range(self.size - 1):
                if self.board[i, j] == self.board[i+1, j]:
                    return False
        return True

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"
        if action == 0:
            moved = self.move_up()
        elif action == 1:
            moved = self.move_down()
        elif action == 2:
            moved = self.move_left()
        elif action == 3:
            moved = self.move_right()
        else:
            moved = False
        self.last_move_valid = moved
        if moved:
            self.add_random_tile()
        done = self.is_game_over()
        return self.board, self.score, done, {}

    def render(self, mode="human", action=None):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(-0.5, self.size - 0.5)
        for i in range(self.size):
            for j in range(self.size):
                value = self.board[i, j]
                color = COLOR_MAP.get(value, "#3c3a32")
                text_color = TEXT_COLOR.get(value, "white")
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color, edgecolor="black")
                ax.add_patch(rect)
                if value != 0:
                    ax.text(j, i, str(value), ha='center', va='center',
                            fontsize=16, fontweight='bold', color=text_color)
        title = f"score: {self.score}"
        if action is not None:
            title += f" | action: {self.actions[action]}"
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.show()

    def simulate_row_move(self, row):
        new_row = row[row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        for i in range(len(new_row) - 1):
            if new_row[i] == new_row[i + 1] and new_row[i] != 0:
                new_row[i] *= 2
                new_row[i + 1] = 0
        new_row = new_row[new_row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        return new_row

    def is_move_legal(self, action):
        temp_board = self.board.copy()
        if action == 0:
            for j in range(self.size):
                col = temp_board[:, j]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col
        elif action == 1:
            for j in range(self.size):
                col = temp_board[:, j][::-1]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col[::-1]
        elif action == 2:
            for i in range(self.size):
                row = temp_board[i]
                temp_board[i] = self.simulate_row_move(row)
        elif action == 3:
            for i in range(self.size):
                row = temp_board[i][::-1]
                new_row = self.simulate_row_move(row)
                temp_board[i] = new_row[::-1]
        else:
            raise ValueError("Invalid action")
        return not np.array_equal(self.board, temp_board)

# Heuristic function to evaluate board states
def evaluate_board(board):
    """
    Evaluate the board state using heuristics:
    - Corner strategy: Reward high tiles in bottom-right (3,3)
    - Monotonicity: Encourage increasing/decreasing tiles along rows/columns
    - Empty tiles: Reward more empty tiles
    - Smoothness: Minimize differences between adjacent tiles
    """
    # Convert tiles to log2 values for smoother differences
    log_board = np.zeros_like(board, dtype=float)
    for i in range(4):
        for j in range(4):
            log_board[i, j] = 0 if board[i, j] == 0 else math.log2(board[i, j])

    # Corner strategy: Reward high tiles in (3,3)
    corner_score = log_board[3, 3] * 1000  # High weight for bottom-right

    # Monotonicity: Prefer increasing tiles towards bottom-right
    mono_score = 0
    # Rows: Prefer increasing rightward or leftward
    for i in range(4):
        row = log_board[i]
        # Increasing rightward
        diffs_right = sum(row[j+1] - row[j] for j in range(3) if row[j] != 0 and row[j+1] != 0)
        # Increasing leftward
        diffs_left = sum(row[j] - row[j+1] for j in range(3) if row[j] != 0 and row[j+1] != 0)
        mono_score += max(diffs_right, diffs_left) * 100  # Reward the better direction
    # Columns: Prefer increasing downward or upward
    for j in range(4):
        col = log_board[:, j]
        # Increasing downward
        diffs_down = sum(col[i+1] - col[i] for i in range(3) if col[i] != 0 and col[i+1] != 0)
        # Increasing upward
        diffs_up = sum(col[i] - col[i+1] for i in range(3) if col[i] != 0 and col[i+1] != 0)
        mono_score += max(diffs_down, diffs_up) * 100

    # Empty tiles: More empty tiles = more flexibility
    empty_tiles = np.count_nonzero(board == 0)
    empty_score = empty_tiles * 500  # High weight to keep board open

    # Smoothness: Minimize differences between adjacent tiles
    smoothness_score = 0
    for i in range(4):
        for j in range(3):
            if board[i, j] != 0 and board[i, j+1] != 0:
                smoothness_score -= abs(log_board[i, j] - log_board[i, j+1]) * 50
    for j in range(4):
        for i in range(3):
            if board[i, j] != 0 and board[i+1, j] != 0:
                smoothness_score -= abs(log_board[i, j] - log_board[i+1, j]) * 50

    # Combine scores
    total_score = corner_score + mono_score + empty_score + smoothness_score
    return total_score

# Simple MCTS class using the heuristic for rollouts
class MCTS:
    class Node:
        def __init__(self, env, state, score, parent=None, action=None):
            self.env = copy.deepcopy(env)
            self.env.board = state.copy()
            self.env.score = score
            self.state = state
            self.score = score
            self.parent = parent
            self.action = action
            self.children = {}
            self.visits = 0
            self.total_value = 0.0
            self.untried_actions = [a for a in range(4) if self.env.is_move_legal(a)]

        def fully_expanded(self):
            return len(self.untried_actions) == 0

        def ucb_score(self, exploration_weight=1.0):
            if self.visits == 0:
                return float('inf')
            exploitation = self.total_value / self.visits
            exploration = exploration_weight * math.sqrt(math.log(self.parent.visits) / self.visits)
            return exploitation + exploration

    def __init__(self, env, iterations=50, rollout_depth=5):
        self.env = env
        self.iterations = iterations
        self.rollout_depth = rollout_depth

    def create_env_from_state(self, state, score):
        new_env = copy.deepcopy(self.env)
        new_env.board = state.copy()
        new_env.score = score
        return new_env

    def select_child(self, node):
        return max(node.children.values(), key=lambda n: n.ucb_score())

    def rollout(self, sim_env, depth):
        rollout_env = copy.deepcopy(sim_env)
        for _ in range(depth):
            if rollout_env.is_game_over():
                break
            legal_moves = [a for a in range(4) if rollout_env.is_move_legal(a)]
            if not legal_moves:
                break
            action = random.choice(legal_moves)
            rollout_env.step(action)
        return evaluate_board(rollout_env.board) + rollout_env.score

    def backpropagate(self, node, value):
        current = node
        while current is not None:
            current.visits += 1
            current.total_value += value
            current = current.parent

    def run_simulation(self, root):
        node = root
        sim_env = self.create_env_from_state(node.state, node.score)
        while node.fully_expanded() and not sim_env.is_game_over():
            node = self.select_child(node)
            sim_env.board = node.state.copy()
            sim_env.score = node.score
        if not node.fully_expanded() and not sim_env.is_game_over():
            action = random.choice(node.untried_actions)
            node.untried_actions.remove(action)
            new_state, new_score, done, _ = sim_env.step(action)
            child = self.Node(sim_env, new_state.copy(), new_score, parent=node, action=action)
            node.children[action] = child
            node = child
        rollout_value = self.rollout(sim_env, self.rollout_depth)
        self.backpropagate(node, rollout_value)

    def best_action(self, root):
        for _ in range(self.iterations):
            self.run_simulation(root)
        if not root.children:
            return None
        return max(root.children.items(), key=lambda item: item[1].visits)[0]

# Initialize environment and MCTS
env = Game2048Env()
mcts = MCTS(env, iterations=50, rollout_depth=5)

def get_action(state, score):
    # Set up the environment with the given state and score
    env.board = state.copy()
    env.score = score

    # Create the root node for MCTS
    root = mcts.Node(env, state.copy(), score)

    # Run MCTS to find the best action
    best_action = mcts.best_action(root)

    # Fallback to heuristic if MCTS fails
    if best_action is None:
        legal_moves = [a for a in range(4) if env.is_move_legal(a)]
        if not legal_moves:
            return random.choice([0, 1, 2, 3])  # Fallback if no legal moves
        best_value = -float('inf')
        best_action = legal_moves[0]
        temp_env = copy.deepcopy(env)
        for action in legal_moves:
            new_state, new_score, _, _ = temp_env.step(action)
            value = evaluate_board(new_state) + new_score
            if value > best_value:
                best_value = value
                best_action = action
            temp_env.board = state.copy()
            temp_env.score = score

    return best_action