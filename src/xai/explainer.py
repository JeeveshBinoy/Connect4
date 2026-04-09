import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.core.game import Connect4

class Explainer:
    def __init__(self, model):
        self.model = model

    def get_saliency_map(self, state):
        """
        Calculates saliency map using gradients of the output (value) w.r.t input board.
        State: (2, 6, 7)
        Returns: (6, 7) heatmap
        """
        self.model.eval()
        input_tensor = torch.FloatTensor(state).unsqueeze(0)
        input_tensor.requires_grad = True
        
        pi, v = self.model(input_tensor)
        
        # We want to see which pieces influence the WIN probability (v)
        v.backward()
        
        # Gradient shape: (1, 2, 6, 7)
        gradients = input_tensor.grad.data.abs().numpy()[0]
        
        # Merge channels (sum or max) to get a board heatmap
        saliency = np.max(gradients, axis=0)
        
        # Normalize for visualization
        if np.max(saliency) > 0:
            saliency = saliency / np.max(saliency)
            
        return saliency

    def get_strategy_explanation(self, state, move, q_values, visit_counts):
        """
        Generates a highly detailed natural language explanation for the chosen move.
        """
        board = (state[0] - state[1]).astype(np.int8)
        col = move
        
        explanation = []
        
        # 1. Immediate Win
        temp_game = Connect4()
        temp_game.board = np.copy(board)
        temp_game.current_player = 1
        _, winner, _ = temp_game.make_move(col)
        if winner != 0:
            return "I played here because it immediately connects four pieces in a row, winning the game. My simulation correctly identified this as the final, deciding move."

        # 2. Defensive Block
        opp_game = Connect4()
        opp_game.board = np.copy(board)
        opp_game.current_player = -1
        _, opp_winner, _ = opp_game.make_move(col)
        if opp_winner != 0:
            return "I was forced to play in this column because you were threatening an immediate win on your next turn. If I had played anywhere else, you would have connected four pieces and ended the game."

        # Narrative generation
        paragraphs = []
        
        # Paragraph 1: Calculation
        total_visits = sum(visit_counts)
        move_visits = visit_counts[col]
        sorted_visits = sorted(visit_counts, reverse=True)
        margin = sorted_visits[0] - sorted_visits[1] if len(sorted_visits) > 1 else 0
        
        paragraphs.append(f"I chose to play in Column {col + 1}. I arrived at this decision after simulating {total_visits} different future board combinations to see how you might respond.")

        # Paragraph 2: Win Probability (Tension/Balance)
        win_prob = (q_values[col] + 1) / 2.0
        if win_prob > 0.90:
            paragraphs.append(f"The underlying math shows that my position is now overwhelmingly strong. I estimate a {(win_prob*100):.1f}% probability that I will win this match from this position.")
        elif win_prob < 0.15:
            paragraphs.append(f"You have backed me into a very difficult corner. My simulations show you have a massive advantage, and I only estimate a {(win_prob*100):.1f}% chance of surviving. I am playing here to block your most dangerous paths and prolong the game.")
        else:
            paragraphs.append(f"The game is currently very balanced, with neither of us having a definitive advantage (my estimated win chance is {(win_prob*100):.1f}%). Every piece placed right now is critical for establishing control.")

        # Paragraph 3: Positional breakdown
        if col == 3:
             paragraphs.append("I chose the exact center column because it is the most valuable real estate on the board. Controlling the center gives me the maximum number of potential horizontal and diagonal connections as the board fills up.")
        elif col in [2, 4]:
             paragraphs.append("By playing in the inner-middle columns, I am keeping my offensive options flexible while trying to prevent you from dominating the center of the board.")
        else:
             paragraphs.append("Playing on the outer edges is a deliberate tactical choice. I am trying to force you to stretch your defenses away from the center, hoping to create an opening where you cannot block multiple threats at once.")
             
        # Paragraph 4: Confidence
        if margin > total_visits * 0.5:
             paragraphs.append("I am highly confident in this move. The mathematical difference between this choice and my second-best option was massive, meaning this is clearly the optimal play.")
        elif margin < total_visits * 0.15:
             paragraphs.append("This was actually a very difficult decision for me. There were several other columns that looked almost equally strong, and I had to simulate deep into the future to find the slight advantage this move provides.")
        
        return "<br><br>".join(paragraphs)

    def get_mcts_insights(self, mcts, state, move):
        """
        Extracts strategic insights from MCTS for a given state.
        """
        s = state.tobytes()
        if s not in mcts.Ps:
            return None
            
        q_values = [float(mcts.Qsa.get((s, a), 0)) for a in range(7)]
        visit_counts = [int(mcts.Nsa.get((s, a), 0)) for a in range(7)]
        explanation = self.get_strategy_explanation(state, move, q_values, visit_counts)
        
        insights = {
            'prior_probs': [float(x) for x in mcts.Ps[s]],
            'visit_counts': [int(mcts.Nsa.get((s, a), 0)) for a in range(7)],
            'q_values': q_values,
            'value_score': float(mcts.Es.get(s, 0) if mcts.Es.get(s, 0) is not None else 0),
            'explanation': explanation
        }
        return insights
