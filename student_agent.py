import numpy as np
import random
import copy
import math

# In this part I want to define colors to render the board, so I'll use dictionaries for tile value color mapping.
COLOR_MAP = {
    0: "#cdc1b4", 2: "#eee4da", 4: "#ede0c8", 8: "#f2b179", 16: "#f59563",
    32: "#f67c5f", 64: "#f65e3b", 128: "#edcf72", 256: "#edcc61", 512: "#edc850",
    1024: "#edc53f", 2048: "#edc22e", 4096: "#3c3a32", 8192: "#3c3a32"
}
TEXT_COLOR = {key: "white" if key >= 8 else "black" for key in COLOR_MAP.keys()}

# In this part I want to simulate the 2048 game logic, so I'll implement Game2048Env with state transitions.
class Game2048Env:
    def __init__(self):
        self.size = 4
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.reset()

    def reset(self):
        self.board.fill(0)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.board

    def add_random_tile(self):
        empties = list(zip(*np.where(self.board == 0)))
        if empties:
            x, y = random.choice(empties)
            self.board[x, y] = 2 if random.random() < 0.9 else 4

    def compress(self, row):
        nonzeros = row[row != 0]
        return np.pad(nonzeros, (0, self.size - len(nonzeros)), mode='constant')

    def merge(self, row):
        for i in range(self.size - 1):
            if row[i] == row[i + 1] and row[i] != 0:
                row[i] *= 2
                row[i + 1] = 0
                self.score += row[i]
        return row

    def move_left(self):
        moved = False
        for i in range(self.size):
            original = self.board[i].copy()
            row = self.compress(self.board[i])
            row = self.merge(row)
            row = self.compress(row)
            self.board[i] = row
            if not np.array_equal(original, row):
                moved = True
        return moved

    def move_right(self):
        moved = False
        for i in range(self.size):
            original = self.board[i].copy()
            row = self.compress(self.board[i][::-1])
            row = self.merge(row)
            row = self.compress(row)
            self.board[i] = row[::-1]
            if not np.array_equal(original, self.board[i]):
                moved = True
        return moved

    def move_up(self):
        moved = False
        for j in range(self.size):
            original = self.board[:, j].copy()
            col = self.compress(self.board[:, j])
            col = self.merge(col)
            col = self.compress(col)
            self.board[:, j] = col
            if not np.array_equal(original, col):
                moved = True
        return moved

    def move_down(self):
        moved = False
        for j in range(self.size):
            original = self.board[:, j].copy()
            col = self.compress(self.board[::-1, j])
            col = self.merge(col)
            col = self.compress(col)
            self.board[:, j] = col[::-1]
            if not np.array_equal(original, self.board[:, j]):
                moved = True
        return moved

    def is_move_legal(self, action):
        clone = copy.deepcopy(self)
        return clone.step(action)[1] != self.score

    def step(self, action):
        moved = False
        if action == 0:
            moved = self.move_up()
        elif action == 1:
            moved = self.move_down()
        elif action == 2:
            moved = self.move_left()
        elif action == 3:
            moved = self.move_right()
        if moved:
            self.add_random_tile()
        done = self.is_game_over()
        return self.board, self.score, done, {}

    def is_game_over(self):
        if np.any(self.board == 0):
            return False
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i, j] == self.board[i, j + 1] or self.board[j, i] == self.board[j + 1, i]:
                    return False
        return True

# In this part I want to evaluate boards based on smart features, so I'll write a heuristic.
def evaluate_board(board):
    log_board = np.where(board > 0, np.log2(board), 0)

    corner_bonus = log_board[3, 3] * 1000  # favor large tile in bottom-right
    empty_tiles = np.count_nonzero(board == 0) * 200

    smoothness = 0
    for i in range(4):
        for j in range(3):
            if board[i, j] and board[i, j+1]:
                smoothness -= abs(log_board[i, j] - log_board[i, j+1]) * 50
            if board[j, i] and board[j+1, i]:
                smoothness -= abs(log_board[j, i] - log_board[j+1, i]) * 50

    return corner_bonus + empty_tiles + smoothness

# In this part I want to do MCTS simulations, so I'll define a lightweight rollout-based MCTS class.
class MCTS:
    def __init__(self, env, iterations=100, rollout_depth=5):
        self.env = env
        self.iterations = iterations
        self.rollout_depth = rollout_depth

    def rollout(self, state, score):
        sim_env = Game2048Env()
        sim_env.board = state.copy()
        sim_env.score = score

        for _ in range(self.rollout_depth):
            legal_moves = [a for a in range(4) if sim_env.is_move_legal(a)]
            if not legal_moves:
                break
            action = random.choice(legal_moves)
            sim_env.step(action)

        return evaluate_board(sim_env.board) + sim_env.score

    def best_action(self, state, score):
        best_val = -float('inf')
        best_act = 0
        for action in range(4):
            if not self.env.is_move_legal(action):
                continue
            sim_env = copy.deepcopy(self.env)
            sim_env.board = state.copy()
            sim_env.score = score
            sim_env.step(action)

            total = 0
            for _ in range(self.iterations):
                total += self.rollout(sim_env.board, sim_env.score)
            avg_val = total / self.iterations

            if avg_val > best_val:
                best_val = avg_val
                best_act = action

        return best_act

# In this part I want to select the best move using MCTS, so I'll use the class above.
def get_action(state, score):
    env = Game2048Env()
    env.board = state.copy()
    env.score = score
    mcts = MCTS(env, iterations=100, rollout_depth=5)
    return mcts.best_action(state, score)
