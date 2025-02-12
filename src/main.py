import sys

MIN_PYTHON_VERSION = (3, 10)

if sys.version_info < MIN_PYTHON_VERSION:
    current_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    sys.stderr.write(f"Python {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]} or later is required. You are using Python {current_version}.\n")
    sys.exit(1)

from game import Game
from solutions.random import RandomSolution
from solutions.dqn import DQNSolution
from solutions.genetic import GeneticSolution
from solutions.minimax import MiniMaxSolution
from solutions.mcts import MCTSSolution
from solutions.dqn_trainer import MoveBlocker
from colorama import Fore, Style, init
import os

SOLUTIONS = {
    "1": ("Deep Q-Network (DQN)", DQNSolution, "DQN"),
    "2": ("Genetic Algorithm", GeneticSolution, "Genetic"),
    "3": ("Minimax", MiniMaxSolution, "Minimax"),
    "4": ("Monte Carlo Tree Search", MCTSSolution, "MCTS"),
    "5": ("Random", RandomSolution, "Random"),
    "6": ("Tactical", MoveBlocker, "Tactical")
}

def print_menu(is_second_agent=False):
    if not is_second_agent:
        init(autoreset=True)
        os.system('cls' if os.name == 'nt' else 'clear')
        print(Fore.CYAN + r"""
  ____                            _     _  _   
 / ___|___  _ __  _ __   ___  ___| |_  | || |  
| |   / _ \| '_ \| '_ \ / _ \/ __| __| | || |_ 
| |__| (_) | | | | | | |  __/ (__| |_  |__   _|
 \____\___/|_| |_|_| |_|\___|\___|\__|    |_|  
        """)
        print(Fore.RED + "Select game mode:")
        print(Fore.GREEN + "1. Human vs AI")
        print(Fore.YELLOW + "2. AI vs AI")
    else:
        print(Fore.RED + "\nSelect second AI:")
    
    if not is_second_agent:
        return

    print(Fore.RED + "\nSelect an algorithm:")
    for key, (name, _, _) in SOLUTIONS.items():
        color = Fore.GREEN if int(key) % 2 == 1 else Fore.YELLOW
        print(f"{color}{key}. {name}")

def get_valid_input(prompt, valid_options):
    while True:
        choice = input(prompt)
        if choice in valid_options:
            return choice
        print(Fore.RED + f"Invalid choice. Please choose from {', '.join(valid_options)}")

def main():
    print_menu()
    mode = get_valid_input("Enter game mode (1-2): ", ["1", "2"])

    if mode == "1":  # Human vs AI
        print_menu(True)
        ai_choice = get_valid_input("Enter the number of your choice: ", SOLUTIONS.keys())
        rounds = 1
        solution = SOLUTIONS[ai_choice][1]

        game = Game(solution, human_vs_ai=True, 
                   player1_label="Human",
                   player2_label=SOLUTIONS[ai_choice][2])
        game.run()
        
    else:  # AI vs AI
        print_menu(True)
        ai1_choice = get_valid_input("Select first AI: ", SOLUTIONS.keys())
        print_menu(True)
        ai2_choice = get_valid_input("Select second AI: ", SOLUTIONS.keys())
        
        rounds = int(get_valid_input("Enter number of rounds (1-100): ", [str(i) for i in range(1, 101)]))
        
        solution1 = SOLUTIONS[ai1_choice][1]
        solution2 = SOLUTIONS[ai2_choice][1]
        
        print(f"\nStarting {rounds} rounds between {SOLUTIONS[ai1_choice][0]} and {SOLUTIONS[ai2_choice][0]}")

        game = Game(solution1, ai_vs_ai=True, opponent_solution=solution2, rounds=rounds,
                   player1_label=SOLUTIONS[ai2_choice][2],
                   player2_label=SOLUTIONS[ai1_choice][2])
        game.run()

if __name__ == "__main__":
    main()