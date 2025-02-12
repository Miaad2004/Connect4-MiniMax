import pygame
import sys
import numpy as np
from constants import *
from solutions.solution import Solution
import random

class Board:
    def __init__(self):
        self.board = np.zeros((ROW_COUNT, COLUMN_COUNT))
    
    def drop_piece(self, row, col, piece):
        self.board[row][col] = piece
    
    def is_valid_location(self, col):
        if col < 0 or col >= COLUMN_COUNT:
            return False
        
        return self.board[ROW_COUNT-1][col] == 0
    
    def get_next_open_row(self, col):
        for r in range(ROW_COUNT):
            if self.board[r][col] == 0:
                return r
                
    def get_valid_locations(self):
        return [col for col in range(COLUMN_COUNT) if self.is_valid_location(col)]
    
    def winning_move(self, piece):
        # Horizontal check
        for c in range(COLUMN_COUNT-3):
            for r in range(ROW_COUNT):
                if all(self.board[r][c+i] == piece for i in range(4)):
                    return True
                    
        # Vertical check
        for c in range(COLUMN_COUNT):
            for r in range(ROW_COUNT-3):
                if all(self.board[r+i][c] == piece for i in range(4)):
                    return True
                    
        # Positive diagonal check
        for c in range(COLUMN_COUNT-3):
            for r in range(ROW_COUNT-3):
                if all(self.board[r+i][c+i] == piece for i in range(4)):
                    return True
                    
        # Negative diagonal check
        for c in range(COLUMN_COUNT-3):
            for r in range(3, ROW_COUNT):
                if all(self.board[r-i][c+i] == piece for i in range(4)):
                    return True
        
        return False

    def is_terminal_node(self):
        return (self.winning_move(PLAYER_PIECE)
                or self.winning_move(AI_PIECE)
                or len(self.get_valid_locations()) == 0)
    
    def get_reward(self):
        if self.winning_move(PLAYER_PIECE):
            return -1000
        
        elif self.winning_move(AI_PIECE):
            return 100
        
        else:
            # Get positional score
            ai_score = self.get_score(AI_PIECE)
            player_score = self.get_score(PLAYER_PIECE)
            return ai_score - player_score
    
    def get_state_key(self, board=None):
        if not board:
            board = self.board
        
        return tuple(tuple(x) for x in board)

    def evaluate_window(self, window, piece):
        score = 0
        opp_piece = PLAYER_PIECE if piece == AI_PIECE else AI_PIECE

        # Winning patterns
        if window.count(piece) == 4:
            score += 100
        # Near wins
        elif window.count(piece) == 3 and window.count(0) == 1:
            score += 5
        # Two pieces with space
        elif window.count(piece) == 2 and window.count(0) == 2:
            score += 2

        # Blocking patterns
        if window.count(opp_piece) == 3 and window.count(0) == 1:
            score -= 4  # Block opponent's potential win

        return score

    def get_score(self, piece):
        score = 0
        
        # Score center column
        center_array = [int(i) for i in list(self.board[:, COLUMN_COUNT//2])]
        center_count = center_array.count(piece)
        score += center_count * 3

        # Horizontal
        for r in range(ROW_COUNT):
            row_array = [int(i) for i in list(self.board[r,:])]
            for c in range(COLUMN_COUNT-3):
                window = row_array[c:c+WINDOW_LENGTH]
                score += self.evaluate_window(window, piece)

        # Vertical
        for c in range(COLUMN_COUNT):
            col_array = [int(i) for i in list(self.board[:,c])]
            for r in range(ROW_COUNT-3):
                window = col_array[r:r+WINDOW_LENGTH]
                score += self.evaluate_window(window, piece)

        # Positive diagonal
        for r in range(ROW_COUNT-3):
            for c in range(COLUMN_COUNT-3):
                window = [self.board[r+i][c+i] for i in range(WINDOW_LENGTH)]
                score += self.evaluate_window(window, piece)

        # Negative diagonal
        for r in range(3, ROW_COUNT):
            for c in range(COLUMN_COUNT-3):
                window = [self.board[r-i][c+i] for i in range(WINDOW_LENGTH)]
                score += self.evaluate_window(window, piece)

        return score


class Game:
    def __init__(self, solution_class, human_vs_ai=False, ai_vs_ai=False, opponent_solution=None, rounds=1, 
                 player1_label="Player", player2_label="AI", ai_move_delay=1):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        
        self.board = Board()
        self.solution: Solution = solution_class(self.board)
        self.opponent_solution = opponent_solution(self.board) if opponent_solution else None
        self.game_over = False
        self.turn = np.random.randint(PLAYER, AI+1)
        
        self.human_vs_ai = human_vs_ai
        self.ai_vs_ai = ai_vs_ai
        self.ai_move_delay = ai_move_delay
        self.rounds = rounds
        self.current_round = 1
        
        self.player_score = 0
        self.ai_score = 0
        
        self.player1_label = player1_label
        self.player2_label = player2_label
        
        self.font = pygame.font.SysFont("monospace", 50)
        self.score_font = pygame.font.SysFont("monospace", 35)
        
    def draw_board(self):
        # Draw board background
        for c in range(COLUMN_COUNT):
            for r in range(ROW_COUNT):
                pygame.draw.rect(self.screen, BLUE, 
                               (c*SQUARESIZE, r*SQUARESIZE+SQUARESIZE, SQUARESIZE, SQUARESIZE))
                pygame.draw.circle(self.screen, BLACK, 
                                (int(c*SQUARESIZE+SQUARESIZE/2), 
                                 int(r*SQUARESIZE+SQUARESIZE+SQUARESIZE/2)), RADIUS)
        
        # Draw pieces
        for c in range(COLUMN_COUNT):
            for r in range(ROW_COUNT):
                if self.board.board[r][c] == PLAYER_PIECE:
                    pygame.draw.circle(self.screen, RED,
                                    (int(c*SQUARESIZE+SQUARESIZE/2),
                                     HEIGHT-int(r*SQUARESIZE+SQUARESIZE/2)), RADIUS)
                elif self.board.board[r][c] == AI_PIECE:
                    pygame.draw.circle(self.screen, YELLOW,
                                    (int(c*SQUARESIZE+SQUARESIZE/2),
                                     HEIGHT-int(r*SQUARESIZE+SQUARESIZE/2)), RADIUS)
        
        self.draw_top_section()
        pygame.display.update()

    def draw_top_section(self):
        pygame.draw.rect(self.screen, BLACK, (0,0, WIDTH, SQUARESIZE))
        player_label = self.score_font.render(f"{self.player1_label}: {self.player_score}", 1, RED)
        ai_label = self.score_font.render(f"{self.player2_label}: {self.ai_score}", 1, YELLOW)
        self.screen.blit(player_label, (40, 10))
        self.screen.blit(ai_label, (WIDTH-250, 10))

    def run(self):
        while self.current_round <= self.rounds:
            self.draw_board()
            
            while not self.game_over:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        sys.exit()
                        
                    if self.human_vs_ai:
                        if event.type == pygame.MOUSEMOTION:
                            self.draw_top_section()
                            pos_x = event.pos[0]
                            if self.turn == PLAYER:
                                pygame.draw.circle(self.screen, RED, (pos_x, int(SQUARESIZE/2)), RADIUS)
                            pygame.display.update()
                            
                        if event.type == pygame.MOUSEBUTTONDOWN and self.turn == PLAYER:
                            col = int(event.pos[0] // SQUARESIZE)
                            self._handle_move(col, PLAYER_PIECE)
                            self.draw_board()
                
                if not self.human_vs_ai or self.turn == AI:
                    if self.ai_vs_ai and self.turn == PLAYER:
                        col = self.opponent_solution.get_move()
                        
                    else:
                        col = self.solution.get_move()
                    
                    piece = AI_PIECE if self.turn == AI else PLAYER_PIECE
                    self._handle_move(col, piece)
                    self.draw_board()
                    pygame.time.wait(self.ai_move_delay)  
                
                if self.game_over:
                    pygame.time.wait(1000)
                    if self.current_round == self.rounds:
                        self._display_end_screen()
                        pygame.time.wait(3000)
                        return
                    
                    else:
                        self.current_round += 1
                        self.game_over = False
                        self.board = Board()
                        self.solution.reset(board=self.board)
                        if self.opponent_solution:
                            self.opponent_solution.reset(board=self.board)
                    
                        self.turn = np.random.randint(PLAYER, AI+1)
                        break

    def _handle_move(self, col, piece):
        # Add check for None
        if col is None:
            print(f"Warning: AI returned None instead of a valid move")
            # Choose random valid move as fallback
            valid_moves = self.board.get_valid_locations()
            if valid_moves:
                col = random.choice(valid_moves)
            else:
                return
                
        if self.board.is_valid_location(col):
            row = self.board.get_next_open_row(col)
            self.board.drop_piece(row, col, piece)
            
            if self.board.winning_move(piece):
                if piece == PLAYER_PIECE:
                    self.player_score += 1
                else:
                    self.ai_score += 1
                self.game_over = True
            # Add check for tie condition
            elif len(self.board.get_valid_locations()) == 0:
                self.game_over = True
                
            self.turn = AI if piece == PLAYER_PIECE else PLAYER
    
    def _display_end_screen(self):
        end_screen = pygame.display.set_mode((500, 300))
        pygame.display.set_caption("Game Over")
    
        end_screen.fill(WHITE)
        title_font = pygame.font.SysFont("monospace", 50)
        score_font = pygame.font.SysFont("monospace", 35)
    
        # Add tie condition
        if self.player_score == self.ai_score:
            winner_label = title_font.render("It's a Tie!", 1, BLUE)
        else:
            winner = self.player1_label if self.player_score > self.ai_score else self.player2_label
            winner_label = title_font.render(f"{winner} Wins!", 1, RED)
    
        player_score_label = score_font.render(f"{self.player1_label} Score: {self.player_score}", 1, BLACK)
        ai_score_label = score_font.render(f"{self.player2_label} Score: {self.ai_score}", 1, BLACK)
    
        end_screen.blit(winner_label, (100, 50))
        end_screen.blit(player_score_label, (100, 150))
        end_screen.blit(ai_score_label, (100, 200))
        pygame.display.update()
    
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()