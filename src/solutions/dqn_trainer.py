from .solution import Solution
import numpy as np
from game import Board
from constants import AI_PIECE, PLAYER_PIECE, ROW_COUNT, COLUMN_COUNT, WINDOW_LENGTH

PLAYER_PIECE2 = AI_PIECE
AI_PIECE2 = PLAYER_PIECE

class MoveBlocker(Solution):
    def __init__(self, board: Board):
        self.board = board
        self.verbose = True
    
    def get_move(self):
        valid_locations = self.board.get_valid_locations()
        if len(valid_locations) == 0:
            return None
        
        # 1: Check for winning move
        for col in valid_locations:
            row = self.board.get_next_open_row(col)
            board_copy = self.board.board.copy()
            if row is not None:
                self.board.drop_piece(row, col, PLAYER_PIECE2)
                if self.board.winning_move(PLAYER_PIECE2):
                    self.board.board = board_copy
                    if self.verbose:
                        print("Winner Winner Chicken Dinner!")
                    return col
                self.board.board = board_copy

        # 2: Block winning move
        for col in valid_locations:
            row = self.board.get_next_open_row(col)
            board_copy = self.board.board.copy()
            if row is not None:
                self.board.drop_piece(row, col, AI_PIECE2)
                if self.board.winning_move(AI_PIECE2):
                    self.board.board = board_copy
                    if self.verbose:
                        print("blocking winning move")
                        
                    return col
                self.board.board = board_copy
                
        # 3: Block horizontal and diagonal threats
        for col in valid_locations:
            row = self.board.get_next_open_row(col)
            if row is not None:
                if self._check_horizontal_threat(row, col, AI_PIECE2) or \
                   self._check_diagonal_threat(row, col, AI_PIECE2):
                    if self.verbose:
                        print("blocking horizontal and diagonal threat")   
                        
                    return col

        # 4: Block vertical stacking
        for col in valid_locations:
            row = self.board.get_next_open_row(col)
            if row is not None and row > 0 and self.board.board[row - 1][col] == AI_PIECE2:
                if self.verbose:
                    print("blocking vertical stacking")
                    
                return col
        
        # 5: random
        return np.random.choice(valid_locations)
    
    def _check_diagonal_threat(self, row, col, piece):
        # positive diag (/)
        for i in range(-3, 1):
            if 0 <= row+i < ROW_COUNT-3 and 0 <= col+i < COLUMN_COUNT-3:
                count = 0
                spaces = []
 
                for j in range(4):
                    curr_row, curr_col = row+i+j, col+i+j
                    if self.board.board[curr_row][curr_col] == piece:
                        count += 1
                    elif self.board.board[curr_row][curr_col] == 0:
                        spaces.append((curr_row, curr_col))
                        
                # If there are 2 pieces and 2 empty spaces in line
                if count == 2 and len(spaces) == 2:
                    # Check if empty spaces are reachable 
                    for space_row, space_col in spaces:
                        if space_row == 0 or self.board.board[space_row-1][space_col] != 0:
                            return True

        # Check negative diag (\)
        for i in range(-3, 1):
            if 0 <= row-i < ROW_COUNT and 0 <= col+i < COLUMN_COUNT-3:
                count = 0
                spaces = []
                
                # Check a window of 4 positions
                for j in range(4):
                    curr_row, curr_col = row-i-j, col+i+j
                    
                    if 0 <= curr_row < ROW_COUNT:  
                        if self.board.board[curr_row][curr_col] == piece:
                            count += 1
                            
                        elif self.board.board[curr_row][curr_col] == 0:
                            spaces.append((curr_row, curr_col))
                            
                # If there are 2 pieces and 2 empty spaces in line
                if count == 2 and len(spaces) == 2:
                    # Check if empty spaces are reachable (have support below)
                    for space_row, space_col in spaces:
                        if space_row == 0 or self.board.board[space_row-1][space_col] != 0:
                            return True
                        
        return False

    def _check_horizontal_threat(self, row, col, piece):
        # Check for horizontal threats 
        for start_col in range(max(0, col-3), min(col+1, COLUMN_COUNT-3)):
            count = 0
            spaces = []
            
            # Check a window of 4 positions
            for j in range(4):
                curr_col = start_col + j
                if self.board.board[row][curr_col] == piece:
                    count += 1
                    
                elif self.board.board[row][curr_col] == 0:
                    # Check if this space is reachable
                    if row == 0 or self.board.board[row-1][curr_col] != 0:
                        spaces.append(curr_col)
                        
            # Threat exists if:
            # 1. There are exactly 2 pieces
            # 2. There are exactly 2 reachable empty spaces
            # 3. The empty spaces are not blocked from above
            if count == 2 and len(spaces) == 2:
                # Additional check: make sure the empty spaces are next to pieces
                pieces_cols = [c for c in range(start_col, start_col+4) 
                            if self.board.board[row][c] == piece]
                
                if abs(pieces_cols[0] - pieces_cols[1]) <= 2:  # Pieces are adjacent or have one space between
                    return True
        return False

    
    def reset(self, board):
        self.board = board