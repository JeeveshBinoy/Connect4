import numpy as np

class Connect4:
    def __init__(self):
        self.rows = 6
        self.cols = 7
        self.board = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.current_player = 1 # 1 or -1

    def reset(self):
        self.board = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.current_player = 1
        return self.get_state()

    def get_valid_moves(self):
        return [c for c in range(self.cols) if self.board[0, c] == 0]

    def make_move(self, col):
        if self.board[0, col] != 0:
            raise ValueError("Invalid move: Column is full.")
        
        # Find the row to place the piece
        for r in range(self.rows - 1, -1, -1):
            if self.board[r, col] == 0:
                self.board[r, col] = self.current_player
                break
        
        winner = self.check_winner()
        is_draw = self.check_draw()
        
        self.current_player *= -1
        
        return self.get_state(), winner, is_draw

    def check_winner(self):
        # Check horizontal, vertical, and diagonals
        for r in range(self.rows):
            for c in range(self.cols):
                player = self.board[r, c]
                if player == 0: continue
                
                # Horizontal
                if c + 3 < self.cols and all(self.board[r, c:c+4] == player):
                    return player
                # Vertical
                if r + 3 < self.rows and all(self.board[r:r+4, c] == player):
                    return player
                # Positive Diagonal
                if r + 3 < self.rows and c + 3 < self.cols and \
                   all(self.board[r+i, c+i] == player for i in range(4)):
                    return player
                # Negative Diagonal
                if r + 3 < self.rows and c - 3 >= 0 and \
                   all(self.board[r+i, c-i] == player for i in range(4)):
                    return player
        return 0

    def check_draw(self):
        return np.all(self.board != 0) and self.check_winner() == 0

    def get_state(self):
        # State for CNN input should be (2, rows, cols)
        # Channel 0: Current player's pieces
        # Channel 1: Opponent's pieces
        state = np.zeros((2, self.rows, self.cols), dtype=np.float32)
        state[0, :, :] = (self.board == self.current_player).astype(np.float32)
        state[1, :, :] = (self.board == -self.current_player).astype(np.float32)
        return state

    def get_canonical_form(self):
        # Canonical form is always from the perspective of the current player
        return self.get_state()

    def get_next_state(self, board, player, action):
        # Static-like helper for MCTS to simulate a move without changing self.board
        temp_game = Connect4()
        temp_game.board = np.copy(board)
        temp_game.current_player = player
        state, winner, draw = temp_game.make_move(action)
        return temp_game.get_state(), temp_game.current_player, winner, draw

    def string_representation(self, state):
        return state.tobytes()
