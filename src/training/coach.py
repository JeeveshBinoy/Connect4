import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from collections import deque
import random
import time
import torch.nn.functional as F
from tqdm import tqdm

from src.core.game import Connect4
from src.core.mcts import MCTS
from src.model.network import Connect4Net

class Coach:
    def __init__(self, game, model, args):
        self.game = game
        self.model = model
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Coach initialized. Using device: {self.device}")
        self.model.to(self.device)
        self.mcts = MCTS(self.game, self.model, self.args)
        self.train_examples = deque([], maxlen=args['maxlenOfQueue'])

    def execute_episode(self):
        train_examples = []
        state = self.game.reset()
        episode_step = 0
        
        while True:
            episode_step += 1
            # temp = 1 for exploration in early moves, 0 for exploitation
            temp = 1 if episode_step < self.args['tempThreshold'] else 0
            pi = self.mcts.get_action_probs(state, temp=temp)
            
            # Save example: (state, pi, None) -> None for winner later
            train_examples.append([state, pi, None])
            
            action = np.random.choice(len(pi), p=pi)
            state, winner, draw = self.game.make_move(action)
            
            if winner != 0 or draw:
                # Game over
                num_moves = episode_step
                return self.assign_rewards(train_examples, winner, draw, num_moves)

    def assign_rewards(self, examples, winner, draw, num_moves):
        # Implementation of Heroic Loss
        results = []
        for i, (state, pi, _) in enumerate(examples):
            # The player who made the move at step i is (i % 2)
            # But in MCTS examples, every state is canonical (current player is always P1)
            # So the outcome for state i is from the perspective of the player who's turn it was.
            
            if draw:
                reward = 0
            else:
                # If winner is 1 and it was player 1's turn (canonical), reward is 1.
                # Since each state in examples was canonical for the player whose turn it was:
                # If i-th move was by the winner, outcome is 1.
                # In Connect 4, turns alternate. 
                # If game ended at move N with winner W, the move N was by W.
                # So state N-1 resulted in win for the player moving in N-1.
                # Relation: (num_moves - 1 - i) % 2 == 0 means same player as winner.
                
                is_winner_turn = (num_moves - 1 - i) % 2 == 0
                if is_winner_turn:
                    reward = 1.0 # Winner always gets full credit for winning
                else:
                    # Heroic Loss: Penalize less for longer games
                    # reward = -1.0 + (num_moves / 42.0) * 0.4
                    reward = -1.0 + (num_moves / 42.0) * self.args.get('heroic_scale', 0.4)
            
            results.append((state, pi, reward))
        return results

    def train(self):
        for i in range(self.args['numIters']):
            print(f"\n=======================")
            print(f"ITERATION {i+1}/{self.args['numIters']}")
            print(f"=======================")
            start_time = time.time()
            
            # 1. Self-Play
            iteration_examples = deque([], maxlen=self.args['maxlenOfQueue'])
            for _ in tqdm(range(self.args['numEps']), desc="Self-play"):
                self.mcts = MCTS(self.game, self.model, self.args) # Reset MCTS
                iteration_examples.extend(self.execute_episode())
            
            self.train_examples.extend(iteration_examples)
            
            # 2. Train Network
            print("\nTraining Network...")
            self.train_network()
            
            # Save checkpoint
            torch.save(self.model.state_dict(), f"checkpoint_{i}.pth")
            
            elapsed = time.time() - start_time
            print(f"-> Iteration {i+1} completed in {elapsed:.1f} seconds.\n")

    def train_network(self):
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=self.args['lr'])
        
        # Batch samples
        examples = list(self.train_examples)
        random.shuffle(examples)
        
        for epoch in range(self.args['epochs']):
            print(f"Epoch {epoch+1}")
            batch_count = int(len(examples) / self.args['batch_size'])
            for i in range(batch_count):
                batch = examples[i*self.args['batch_size'] : (i+1)*self.args['batch_size']]
                states, pis, vs = zip(*batch)
                
                states = torch.FloatTensor(np.array(states)).to(self.device)
                target_pis = torch.FloatTensor(np.array(pis)).to(self.device)
                target_vs = torch.FloatTensor(np.array(vs)).unsqueeze(1).to(self.device)
                
                # Forward
                out_pi, out_v = self.model(states)
                
                # Loss
                l_pi = -torch.sum(target_pis * out_pi) / target_pis.size(0)
                l_v = F.mse_loss(out_v, target_vs)
                total_loss = l_pi + l_v
                
                # Backward
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
            
            print(f"Epoch {epoch+1}/{self.args['epochs']} - Loss Pi: {l_pi:.4f}, Loss V: {l_v:.4f}, Total: {total_loss:.4f}")

if __name__ == "__main__":
    game = Connect4()
    model = Connect4Net()
    
    args = {
        'numIters': 20,
        'numEps': 100,
        'tempThreshold': 15,
        'maxlenOfQueue': 200000,
        'numMCTSSims': 100,
        'cpuct': 1.0,
        'dirichlet_alpha': 1.0, # Noise for Connect 4 exploration
        'checkpoint': './temp/',
        'load_model': False,
        'load_folder_file': ('/dev/models/8x8/10000eps/Checkpoint_100', 'best.pth.tar'),
        'lr': 0.001,
        'batch_size': 512,
        'epochs': 20,
        'heroic_scale': 0.0  # PURE win-maximization for an undefeatable agent
    }
    
    coach = Coach(game, model, args)
    print("Starting training with Pure Win-Maximization Rewards (Undefeatable Mode)...")
    coach.train()
    print("Training complete. Saving final model to checkpoint.pth")
    torch.save(model.state_dict(), "checkpoint.pth")
