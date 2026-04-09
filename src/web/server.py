from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import torch
import os
import logging
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from src.core.game import Connect4
from src.core.mcts import MCTS
from src.model.network import Connect4Net
from src.xai.explainer import Explainer

app = FastAPI()

# Load Model
model = Connect4Net()
# Load checkpoint if exists
if os.path.exists("checkpoint.pth"):
    model.load_state_dict(torch.load("checkpoint.pth"))
model.eval()

explainer = Explainer(model)
game = Connect4()
args = {'cpuct': 1.0, 'numMCTSSims': 800}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Serve Frontend
@app.get("/")
async def read_index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))

class MoveRequest(BaseModel):
    board: list # 6x7 list of 0, 1, -1
    player: int # 1 or -1

def check_1_ply(board_np, current_player):
    """
    Check if the current player can win immediately.
    Then check if the opponent can win immediately (block).
    Returns the column to play, or None.
    """
    temp_game = Connect4()
    
    # 1. Can we win?
    for c in range(7):
        if board_np[0, c] == 0:
            temp_game.board = np.copy(board_np)
            temp_game.current_player = current_player
            _, winner, _ = temp_game.make_move(c)
            if winner == current_player:
                return c
                
    # 2. Can we block the opponent from winning?
    for c in range(7):
        if board_np[0, c] == 0:
            temp_game.board = np.copy(board_np)
            temp_game.current_player = -current_player
            _, winner, _ = temp_game.make_move(c)
            if winner == -current_player:
                return c
                
    return None

@app.post("/api/move")
async def get_move(request: MoveRequest):
    try:
        # Setup game state
        board_np = np.array(request.board, dtype=np.int8)
        game.board = board_np
        game.current_player = request.player
        
        canonical_state = game.get_canonical_form()
        
        # Run MCTS anyway to generate XAI insights and neural heatmap
        mcts = MCTS(game, model, args)
        probs = mcts.get_action_probs(canonical_state, temp=0)
        
        # 1-Ply Check: Bulletproof override for immediate threats
        forced_move = check_1_ply(board_np, request.player)
        if forced_move is not None:
            best_move = int(forced_move)
            # Override probabilities to show 100% confidence on UI
            probs = [1.0 if c == best_move else 0.0 for c in range(7)]
        else:
            best_move = int(np.argmax(probs))
        
        # Execute move for AI
        _, winner, draw = game.make_move(best_move)
        
        # XAI Insights
        saliency = explainer.get_saliency_map(canonical_state)
        insights = explainer.get_mcts_insights(mcts, canonical_state, best_move)
        
        # Thought Tree (Principal Variation)
        pv = mcts.get_principal_variation(canonical_state, max_depth=5)
        
        return {
            "move": int(best_move),
            "probs": [float(x) for x in probs],
            "saliency": saliency.tolist(),
            "insights": insights,
            "winner": int(winner),
            "draw": bool(draw),
            "gameOver": bool(winner != 0 or draw),
            "pv": pv
        }
    except Exception as e:
        logger.exception("Error in /api/move")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/reset")
async def reset():
    game.reset()
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
