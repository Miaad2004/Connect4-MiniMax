from .solution import Solution
import numpy as np
from game import Board

class RandomSolution(Solution):
    def __init__(self, board: Board):
        self.board = board
    
    def get_move(self):
        valid_locations = self.board.get_valid_locations()
        move = np.random.choice(valid_locations)
        #print(f"Random move chosen: {move}")
        return move
    
    def reset(self, board):
        self.board = board
