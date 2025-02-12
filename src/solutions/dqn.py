from .solution import Solution
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from tqdm import tqdm
from constants import AI_PIECE, PLAYER_PIECE, ROW_COUNT, COLUMN_COUNT
from game import Board
from .genetic import GeneticSolution
from .minimax import MiniMaxSolution
from .dqn_trainer import MoveBlocker
import copy

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.flat_size = 64 * ROW_COUNT * COLUMN_COUNT
        
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flat_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, COLUMN_COUNT)
        )

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(0) 
            
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  
            
        x = self.conv_layers(x)
        x = x.view(-1, self.flat_size)
        return self.fc_layers(x)
    

class DQNSolution(Solution):
    def __init__(self, board):
        self.board = board
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print("GPU acceleration enabled.")
        
        # hyperparamس
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.0001
        self.max_grad_norm = 0.5
        self.n_episodes = 25000
        self.batch_size = 64
        self.stacking_penalty = -0.3
        
        self.best_reward = float('-inf')
        
        # init networks
        self.policy_net = DQN().to(self.device)
        self.target_net = DQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        self.training = True
        try:
            self.load_model()
            self.training = False
        except:
            print("No saved model found. Starting training...")
            self.train()
        
    def get_state(self, board=None):
        if board is None:
            board = self.board.board
            
        # convert to tensor 
        state = torch.FloatTensor(board)
        return state.to(self.device)

    def choose_action(self, state):
        valid_moves = self.board.get_valid_locations()
        
        if not valid_moves:
            print("DQN: Warning - No valid moves available.")
            return None
            
        # explore
        if self.training and random.random() < self.epsilon:
            return random.choice(valid_moves)
            
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)  # Add batch dimension
            q_values = self.policy_net(state).squeeze()
            
            # add random noise to q_values 
            noise = torch.randn_like(q_values) * 0.01
            q_values = q_values + noise
            
            # mask for invalid moves
            mask = torch.ones(COLUMN_COUNT, device=self.device) * float('-inf')
            mask[valid_moves] = 0
            masked_q_values = q_values + mask
            
            # Get top k actions and randomly select
            k = min(3, len(valid_moves))  
            top_k_values, top_k_indices = masked_q_values.topk(k)
            
            # If not training, select the best action
            if not self.training:
                move = top_k_indices[0].item()
            
            # Random selection from top k 
            else:
                move = top_k_indices[random.randint(0, k-1)].item()
            
            if not self.training:
                print(f"Valid moves: {valid_moves}")
                print(f"Q-values: {q_values}")
                print(f"DQN: Selected move {move} from top {k} moves {top_k_indices.tolist()}")
            
            return move
    
    def get_stacking_penalty(self, action):
        column = [self.board.board[row][action] for row in range(ROW_COUNT)]
        piece_count = sum(1 for piece in column if piece != 0)
        
        return self.stacking_penalty * (piece_count ** 2)

    def train(self):
        episode_rewards = []
        
        mcts_opponent = MoveBlocker(copy.deepcopy(self.board))
        mcts_opponent.verbose = False
        
        pbar = tqdm(range(self.n_episodes))
        for episode in pbar:
            self.board = Board()
            mcts_opponent.reset(copy.deepcopy(self.board))
            state = self.get_state()
            total_reward = 0
            done = False
            
            while not done:
                # Agent's turn
                action = self.choose_action(state)
                if action is None:
                    break
                    
                # Make move and get next state
                row = self.board.get_next_open_row(action)
                self.board.drop_piece(row, action, AI_PIECE)
                
                # Add stacking penalty to reward
                base_reward = self.board.get_reward() / 100.0
                stacking_penalty = self.get_stacking_penalty(action)
                reward = base_reward + stacking_penalty
                
                next_state = self.get_state()
                done = self.board.is_terminal_node()
                
                self.update_model(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
                
                if done:
                    break
                    
                # Opponent's move
                mcts_opponent.reset(copy.deepcopy(self.board))
                opp_action = mcts_opponent.get_move()
                if opp_action is None:
                    break
                    
                row = self.board.get_next_open_row(opp_action)
                self.board.drop_piece(row, opp_action, PLAYER_PIECE)
                
                state = self.get_state()
                done = self.board.is_terminal_node()
            
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            episode_rewards.append(total_reward)
            pbar.set_description(f"Episode {episode+1}/{self.n_episodes} | ε={self.epsilon:.3f} | Reward: {total_reward:.2f}")
        
        self.save_model()
        self.training = False
        print("Training complete!")

    def update_model(self, state, action, reward, next_state, done):
        state = state.unsqueeze(0)
        next_state = next_state.unsqueeze(0)
        action = torch.LongTensor([[action]]).to(self.device)
        reward = torch.FloatTensor([reward]).to(self.device)
        done = torch.FloatTensor([done]).to(self.device)
        
        # current Q value
        current_q_value = self.policy_net(state).gather(1, action)
        
        # target Q value
        with torch.no_grad():
            next_q_value = self.target_net(next_state).max(1)[0].unsqueeze(1)
            target_q_value = reward + (1 - done) * self.gamma * next_q_value
            
        # loss and update
        loss = nn.MSELoss()(current_q_value, target_q_value)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # update target net
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def get_move(self):
        state = self.get_state()
        return self.choose_action(state)

    def save_model(self, path='dqn_model.pth'):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
        }, path)
        
    def load_model(self, path='dqn_model.pth'):
        checkpoint = torch.load(path, weights_only=True)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.epsilon = self.epsilon_min
    
    def reset(self, board):
        self.board = board