import math
import numpy as np
import torch
from .game import Connect4

class MCTS:
    def __init__(self, game, model, args):
        self.game = game
        self.model = model
        self.args = args # { 'cpuct': 1.0, 'numMCTSSims': 100 }
        self.Qsa = {} # stores Q values for (s, a)
        self.Nsa = {} # stores visit counts for (s, a)
        self.Ns = {} # stores visit counts for state s
        self.Ps = {} # stores initial policy (returned by neural net)
        self.Es = {} # stores game.check_winner() (is game over?)
        self.Vs = {} # stores game.get_valid_moves()

    def get_action_probs(self, state, temp=1):
        """
        State: canonical form (2, 6, 7)
        Returns: probability distribution over columns
        """
        # Expand root node to get Ps
        self.search(state)
        s = self.game.string_representation(state)

        # Add Dirichlet Noise for exploration
        if temp > 0 and self.args.get('dirichlet_alpha', 0) > 0:
            valid_moves = self.Vs.get(s, [])
            if valid_moves:
                noise = np.random.dirichlet([self.args['dirichlet_alpha']] * len(valid_moves))
                for i, a in enumerate(valid_moves):
                    self.Ps[s][a] = 0.75 * self.Ps[s][a] + 0.25 * noise[i]

        for i in range(self.args['numMCTSSims'] - 1):
            self.search(state)

        s = state.tobytes()
        counts = [self.Nsa.get((s, a), 0) for a in range(self.game.cols)]

        if temp == 0:
            best_a = np.argmax(counts)
            probs = [0] * len(counts)
            probs[best_a] = 1
            return probs

        counts = [x**(1./temp) for x in counts]
        probs = [x/float(sum(counts)) for x in counts]
        return probs

    def search(self, state, current_player=1):
        """
        Recursive MCTS search
        Returns: negative of the value (for the other player)
        """
        s = self.game.string_representation(state)

        # Check terminal state
        # In a real implementation, we'd need the game board directly or pass it
        # Let's assume the 'state' passed is the canonical state.
        # We need to know if the game ended at this state.
        
        # Base case: Terminal state
        # (This is a bit complex in a stateless-ish MCTS, 
        # but let's assume we have the winner/draw info)
        
        # Simplified: Check if state is in Es
        if s not in self.Es:
            # We need to check the game board in this state. 
            # Board is (2,6,7). P1 pieces are channel 0, P2 are channel 1.
            # Convert back to board with 1s and -1s.
            board = (state[0] - state[1]).astype(np.int8)
            temp_game = Connect4()
            temp_game.board = board
            winner = temp_game.check_winner()
            draw = temp_game.check_draw()
            if winner != 0: self.Es[s] = winner # 1 or -1
            elif draw: self.Es[s] = 0.0001 # Draw
            else: self.Es[s] = None # Not over
            
        if self.Es[s] is not None:
            return -self.Es[s] * current_player 

        if s not in self.Ps:
            # Leaf node: Expand and Evaluate
            device = next(self.model.parameters()).device
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            self.model.eval()
            with torch.no_grad():
                pi, v = self.model(state_tensor)
            
            self.Ps[s] = torch.exp(pi).detach().cpu().numpy()[0]
            # Mask invalid moves
            valid_moves = [c for c in range(self.game.cols) if state[0, 0, c] + state[1, 0, c] == 0]
            mask = np.zeros(self.game.cols)
            mask[valid_moves] = 1
            self.Ps[s] = self.Ps[s] * mask
            sum_ps = np.sum(self.Ps[s])
            if sum_ps > 0:
                self.Ps[s] /= sum_ps
            else:
                self.Ps[s] = mask / np.sum(mask)
            
            self.Vs[s] = valid_moves
            self.Ns[s] = 0
            return -v.item()

        # Selection
        best_u = -float('inf')
        best_a = -1
        valid_moves = self.Vs[s]

        for a in valid_moves:
            if (s, a) in self.Qsa:
                u = self.Qsa[(s, a)] + self.args['cpuct'] * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
            else:
                u = self.args['cpuct'] * self.Ps[s][a] * math.sqrt(self.Ns[s] + 1e-8)
            
            if u > best_u:
                best_u = u
                best_a = a

        a = best_a
        # Simulate move
        board = (state[0] - state[1]).astype(np.int8)
        next_board = np.copy(board)
        # Find row
        for r in range(self.game.rows-1, -1, -1):
            if next_board[r, a] == 0:
                next_board[r, a] = 1 # Always 1 because it's canonical
                break
        
        # Flip perspective for next player
        next_state = np.zeros((2, 6, 7), dtype=np.float32)
        next_state[0] = (next_board == -1).astype(np.float32) # Current P2 is now P1
        next_state[1] = (next_board == 1).astype(np.float32)

        v = self.search(next_state, -current_player)

        # Depth Discount: force AI to win as fast as possible and lose as slowly as possible
        v = v * 0.99

        # Backpropagation
        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1
            
        return -v

    def get_principal_variation(self, canonical_state, max_depth=5):
        """
        Extracts the anticipated line of play (Principal Variation) from the MCTS tree.
        canonical_state is from the perspective of the current player.
        """
        pv = []
        can_state = np.copy(canonical_state)
        
        for _ in range(max_depth):
            s = can_state.tobytes()
            
            if s not in self.Ps:
                break
                
            best_a = None
            max_v = -1
            for a in range(7):
                v_count = self.Nsa.get((s, a), 0)
                if v_count > max_v:
                    max_v = v_count
                    best_a = a
                    
            if best_a is None or max_v == 0:
                break
                
            pv.append(int(best_a))
            
            # Apply move. The active player is always channel 0.
            empty = (can_state[0] + can_state[1] == 0)
            placed = False
            for r in range(5, -1, -1):
                if empty[r, best_a]:
                    can_state[0, r, best_a] = 1.0
                    placed = True
                    break
                    
            if not placed:
                break
                
            # Flip perspective for next player simulator
            next_state = np.zeros_like(can_state)
            next_state[0] = can_state[1]
            next_state[1] = can_state[0]
            can_state = next_state
            
        return pv
