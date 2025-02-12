from abc import ABC, abstractmethod

class Solution(ABC):
    @abstractmethod
    def __init__(self, board):
        pass

    @abstractmethod
    def get_move(self):
        pass
    
    @abstractmethod
    def reset(self, board):
        pass