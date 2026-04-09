import torch
import numpy as np
from src.core.game import Connect4
from src.core.mcts import MCTS
from src.model.network import Connect4Net

def test_mcts():
    game = Connect4()
    model = Connect4Net()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    args = {
        'numMCTSSims': 100,
        'cpuct': 1.0
    }
    
    mcts = MCTS(game, model, args)
    state = game.reset()
    
    print("Starting MCTS simulations...")
    try:
        probs = mcts.get_action_probs(state)
        print("Probs:", probs)
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_mcts()
