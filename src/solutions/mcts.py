from .solution import Solution
import numpy as np
import math
from constants import AI_PIECE, PLAYER_PIECE
import random
from copy import deepcopy

class MCTSNode:
    def __init__(self, board, parent=None, move=None):
        self.board = board
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = board.get_valid_locations()

    def uct_select_child(self, c_param=1.4):
        # Select child 
        return max(self.children, key=lambda c: c.wins/c.visits + 
                  c_param * math.sqrt(2*math.log(self.visits)/c.visits))

    def add_child(self, move, board):
        # new child 
        child = MCTSNode(board, self, move)
        self.untried_moves.remove(move)
        self.children.append(child)
        return child

class MCTSSolution(Solution):
    def __init__(self, board):
        self.board = board
        self.simulation_count = 500  # n iterations

    def get_move(self):
        root = MCTSNode(deepcopy(self.board))

        for _ in range(self.simulation_count):
            node = root
            board_state = deepcopy(self.board)

            # Selection
            while node.untried_moves == [] and node.children != []:
                node = node.uct_select_child()
                if node.move is not None:
                    row = board_state.get_next_open_row(node.move)
                    board_state.drop_piece(row, node.move, AI_PIECE)

            # Expansion
            if node.untried_moves:
                move = random.choice(node.untried_moves)
                row = board_state.get_next_open_row(move)
                board_state.drop_piece(row, move, AI_PIECE)
                node = node.add_child(move, board_state)

            # Simulation
            while not board_state.is_terminal_node():
                valid_moves = board_state.get_valid_locations()
                if not valid_moves:
                    break
                
                move = random.choice(valid_moves)
                row = board_state.get_next_open_row(move)
                piece = AI_PIECE if random.random() < 0.5 else PLAYER_PIECE
                board_state.drop_piece(row, move, piece)

            # Backpropagation
            while node:
                node.visits += 1
                if board_state.winning_move(AI_PIECE):
                    node.wins += 1
                    
                node = node.parent

        # Select best move
        return max(root.children, key=lambda c: c.visits).move

    def _simulate_game(self, board_state):
        while not board_state.is_terminal_node():
            valid_moves = board_state.get_valid_locations()
            if not valid_moves:
                return 0
            
            # winning or blocking moves FIRST (heuristic)
            move = self._get_smart_move(board_state, valid_moves)
            row = board_state.get_next_open_row(move)
            piece = AI_PIECE if random.random() < 0.5 else PLAYER_PIECE
            board_state.drop_piece(row, move, piece)

        if board_state.winning_move(AI_PIECE):
            return 1
        
        elif board_state.winning_move(PLAYER_PIECE):
            return -1
        
        return 0

    def _get_smart_move(self, board_state, valid_moves):
        for move in valid_moves:
            row = board_state.get_next_open_row(move)
            temp_board = deepcopy(board_state)
            temp_board.drop_piece(row, move, AI_PIECE)
            
            # Prioritize a winning move
            if temp_board.winning_move(AI_PIECE):  
                return move
            
            temp_board = deepcopy(board_state)
            temp_board.drop_piece(row, move, PLAYER_PIECE)
            
            # Block opponent's win
            if temp_board.winning_move(PLAYER_PIECE):  
                return move
            
        return random.choice(valid_moves)  
    
    def reset(self, board):
        self.board = board
