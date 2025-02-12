from .solution import Solution
from constants import AI_PIECE, PLAYER_PIECE, WINDOW_LENGTH, COLUMN_COUNT, ROW_COUNT
import numpy as np
import logging
from typing import List, Tuple, Optional
import math

class MiniMaxSolution(Solution):
    def __init__(self, board) -> None:
        self.board = board
        self.DEPTH = 4  
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

    def score_window(self, window: List[int], piece: int) -> int:
        score = 0
        opp_piece = PLAYER_PIECE if piece == AI_PIECE else AI_PIECE

        # count pieces
        piece_count = window.count(piece)
        empty_count = window.count(0)
        opp_count = window.count(opp_piece)

        # scores
        if piece_count == 4:
            score += 100
        elif piece_count == 3 and empty_count == 1:
            score += 5
        elif piece_count == 2 and empty_count == 2:
            score += 2

        if opp_count == 3 and empty_count == 1:
            score -= 4

        return score

    def evaluate_board(self, board: np.ndarray, piece: int) -> int:
        score = 0

        # Score center column
        center_array = [int(i) for i in list(board[:, COLUMN_COUNT//2])]
        center_count = center_array.count(piece)
        score += center_count * 3

        # horizontal
        for r in range(ROW_COUNT):
            row_array = [int(i) for i in list(board[r,:])]
            for c in range(COLUMN_COUNT-3):
                window = row_array[c:c+WINDOW_LENGTH]
                score += self.score_window(window, piece)

        # vertical
        for c in range(COLUMN_COUNT):
            col_array = [int(i) for i in list(board[:,c])]
            for r in range(ROW_COUNT-3):
                window = col_array[r:r+WINDOW_LENGTH]
                score += self.score_window(window, piece)

        # positive diagonal
        for r in range(ROW_COUNT-3):
            for c in range(COLUMN_COUNT-3):
                window = [board[r+i][c+i] for i in range(WINDOW_LENGTH)]
                score += self.score_window(window, piece)

        # negative diagonal
        for r in range(3, ROW_COUNT):
            for c in range(COLUMN_COUNT-3):
                window = [board[r-i][c+i] for i in range(WINDOW_LENGTH)]
                score += self.score_window(window, piece)

        return score

    def minimax(self,
                board,
                is_min_player: bool,
                alpha: float,
                beta: float,
                remaining_depth: int) -> Tuple[Optional[int], float]:

        terminal = self.board.is_terminal_node()
                
        # Base 
        if terminal:
            if self.board.winning_move(AI_PIECE):
                return (None, math.inf)
            
            elif self.board.winning_move(PLAYER_PIECE):
                return (None, -math.inf)
            
            else: 
                return (None, 0)
                
        if remaining_depth == 0:  
            return (None, self.evaluate_board(board.board, AI_PIECE))
        
        # Minimax
        valid_cols = self.board.get_valid_locations()
        
        if is_min_player:  
            best_utility = math.inf
            best_col = np.random.choice(valid_cols)
            
            for col in valid_cols:
                # copy board
                board_copy = self.board.board.copy()
                
                # drop piece
                row = self.board.get_next_open_row(col)
                self.board.drop_piece(row, col, PLAYER_PIECE)
                
                # recur
                _, new_utility = self.minimax(self.board, True, alpha, beta, remaining_depth - 1)
                
                # restore
                self.board.board = board_copy
                
                # update
                if new_utility < best_utility:
                    best_utility = new_utility
                    best_col = col
                
                # prune
                beta = min(beta, best_utility)
                if alpha >= beta:
                    break
                    
            return best_col, best_utility
        
        else:   # MAX
            best_utility = -math.inf
            best_col = np.random.choice(valid_cols)
            
            for col in valid_cols:
                # copy board
                board_copy = self.board.board.copy()
                
                # drop piece
                row = self.board.get_next_open_row(col)
                self.board.drop_piece(row, col, AI_PIECE)
                
                # recur
                _, new_utility = self.minimax(self.board, True, alpha, beta, remaining_depth - 1)
                self.board.board = board_copy
                
                # update score
                if new_utility > best_utility:
                    best_utility = new_utility
                    best_col = col
                
                # prune
                alpha = max(alpha, best_utility)
                if alpha >= beta:
                    break
                    
            return best_col, best_utility

    def get_move(self) -> int:
        col, _ = self.minimax(self.board, self.DEPTH, -math.inf, math.inf, True)
        self.logger.debug(f"Minimax move chosen: {col}")
        return col
    
    def reset(self, board) -> None:
        self.board = board