from .solution import Solution
import numpy as np
from constants import AI_PIECE, PLAYER_PIECE, ROW_COUNT, COLUMN_COUNT
from copy import deepcopy
import random

class Individual:
    def __init__(self, weights=None):
        if weights is None:
            self.weights = np.random.uniform(-1, 1, size=6)
        else:
            self.weights = np.array(weights)
        self.fitness = 0

    def evaluate_position(self, board, col):
        try:
            features = [
                self._center_control(board, col),
                self._vertical_strength(board, col),
                self._horizontal_strength(board, col),
                self._diagonal_strength(board, col),
                self._blocking_score(board, col),
                self._winning_move_score(board, col)
            ]
            return np.sum(np.multiply(features, self.weights))
        
        except Exception as e:
            print(f"Error evaluating position: {e}")
            return float('-inf')

    def _center_control(self, board, col):
        return 1.0 - (abs(col - (COLUMN_COUNT-1)/2) / ((COLUMN_COUNT-1)/2))

    def _vertical_strength(self, board, col):
        row = board.get_next_open_row(col)
        
        if row is None:
            return -1
        
        count = 0
        
        for i in range(1, 4):
            if row - i >= 0 and board.board[row-i][col] == AI_PIECE:
                count += 1
                
        return count / 3 if count > 0 else 0

    def _horizontal_strength(self, board, col):
        row = board.get_next_open_row(col)
        
        if row is None:
            return -1
        
        count = 0
        for c in range(max(0, col-3), min(COLUMN_COUNT, col+4)):
            if c != col and 0 <= c < COLUMN_COUNT and board.board[row][c] == AI_PIECE:
                count += 1
                
        return count / 6 if count > 0 else 0

    def _diagonal_strength(self, board, col):
        row = board.get_next_open_row(col)
        if row is None:
            return -1
        
        count = 0
        for i in range(-3, 4):
            if 0 <= row + i < ROW_COUNT and 0 <= col + i < COLUMN_COUNT:
                if board.board[row+i][col+i] == AI_PIECE:
                    count += 1
                    
            if 0 <= row - i < ROW_COUNT and 0 <= col + i < COLUMN_COUNT:
                if board.board[row-i][col+i] == AI_PIECE:
                    count += 1
                    
        return count / 12 if count > 0 else 0

    def _blocking_score(self, board, col):
        try:
            test_board = deepcopy(board)
            row = test_board.get_next_open_row(col)
            if row is None:
                return -1
            test_board.drop_piece(row, col, PLAYER_PIECE)
            return 1.0 if test_board.winning_move(PLAYER_PIECE) else 0.0
        
        except Exception:
            return 0.0

    def _winning_move_score(self, board, col):
        try:
            test_board = deepcopy(board)
            row = test_board.get_next_open_row(col)
            if row is None:
                return -1
            
            test_board.drop_piece(row, col, AI_PIECE)
            return 1.0 if test_board.winning_move(AI_PIECE) else 0.0
        
        except Exception:
            return 0.0

class GeneticSolution(Solution):
    def __init__(self, board):
        super().__init__(board)
        self.board = board
        self.population_size = 50
        self.tournament_size = 5
        self.mutation_rate = 0.1
        self.generations = 50
        self.population = [Individual() for _ in range(self.population_size)]
        self.best_individual = None
        
        try:
            self.train()
            
        except Exception as e:
            print(f"Error when training: {e}")
            self.best_individual = self.population[0]

    def train(self):
        print("Starting genetic training...")
        best_fitness = float('-inf')
        
        for generation in range(self.generations):
            self._evaluate_population()
            current_best = max(self.population, key=lambda x: x.fitness)
            if current_best.fitness > best_fitness:
                best_fitness = current_best.fitness
                self.best_individual = current_best
            
            new_population = [self.best_individual]
            
            while len(new_population) < self.population_size:
                parent1 = self._tournament_select()
                parent2 = self._tournament_select()
                child1, child2 = self._crossover(parent1, parent2)
                self._mutate(child1)
                self._mutate(child2)
                new_population.extend([child1, child2])
            
            self.population = new_population[:self.population_size]
            
            if generation % 10 == 0:
                print(f"Generation {generation}, Best Fitness: {best_fitness}")

    def _evaluate_population(self):
        for individual in self.population:
            individual.fitness = self._evaluate_fitness(individual)

    def _evaluate_fitness(self, individual):
        test_board = deepcopy(self.board)
        total_score = 0
        
        for _ in range(5):
            valid_moves = test_board.get_valid_locations()
            if not valid_moves:
                break
            
            try:
                move_scores = [individual.evaluate_position(test_board, move) 
                             for move in valid_moves]
                best_move = valid_moves[np.argmax(move_scores)]
                
                row = test_board.get_next_open_row(best_move)
                if row is not None:
                    test_board.drop_piece(row, best_move, AI_PIECE)
                    
                    if test_board.winning_move(AI_PIECE):
                        total_score += 100
                    total_score += test_board.get_reward()
                    
            except Exception as e:
                print(f"Error in fitness evaluation: {e}")
                
        return total_score

    def _tournament_select(self):
        tournament = random.sample(self.population, min(self.tournament_size, len(self.population)))
        return max(tournament, key=lambda x: x.fitness)

    def _crossover(self, parent1, parent2):
        child1_weights = []
        child2_weights = []
        
        for w1, w2 in zip(parent1.weights, parent2.weights):
            if random.random() < 0.5:
                child1_weights.append(w1)
                child2_weights.append(w2)
                
            else:
                child1_weights.append(w2)
                child2_weights.append(w1)
                
        return (Individual(np.array(child1_weights)), 
                Individual(np.array(child2_weights)))

    def _mutate(self, individual):
        if random.random() < self.mutation_rate:
            mutation = np.random.normal(0, 0.1, size=len(individual.weights))
            individual.weights += mutation
            individual.weights = np.clip(individual.weights, -1, 1)

    def get_move(self):
        try:
            valid_moves = self.board.get_valid_locations()
            if not valid_moves:
                return valid_moves[0]
            
            if self.best_individual is None:
                self.best_individual = self.population[0]
                
            move_scores = [self.best_individual.evaluate_position(self.board, move) 
                          for move in valid_moves]
            return valid_moves[np.argmax(move_scores)]
        
        except Exception as e:
            print(f"Error getting move: {e}")
            return valid_moves[0] if valid_moves else 0
        
    def reset(self, board) -> None:
        self.board = board