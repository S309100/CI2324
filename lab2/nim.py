import logging
from pprint import pprint, pformat
from collections import namedtuple
import random
from copy import deepcopy



Nimply = namedtuple("Nimply", "row, num_objects")


class Nim:
    def __init__(self, num_rows: int, k: int = None) -> None:
        self._rows = [i * 2 + 1 for i in range(num_rows)]
        self._k = k

    def __bool__(self):
        return sum(self._rows) > 0

    def __str__(self):
        return "<" + " ".join(str(_) for _ in self._rows) + ">"

    @property
    def rows(self) -> tuple:
        return tuple(self._rows)

    def nimming(self, ply: Nimply) -> None:
        row, num_objects = ply
        assert self._rows[row] >= num_objects
        assert self._k is None or num_objects <= self._k
        self._rows[row] -= num_objects


def pure_random(state: Nim) -> Nimply:
    """A completely random move"""
    row = random.choice([r for r, c in enumerate(state.rows) if c > 0])
    num_objects = random.randint(1, state.rows[row])
    return Nimply(row, num_objects)



def gabriele(state: Nim) -> Nimply:
    """Pick always the maximum possible number of the lowest row"""
    possible_moves = [(r, o) for r, c in enumerate(state.rows) for o in range(1, c + 1)]
    return Nimply(*max(possible_moves, key=lambda m: (-m[0], m[1])))



def adaptive(state: Nim) -> Nimply:
    """A strategy that can adapt its parameters"""
    genome = {"love_small": 0.5}



import numpy as np


def nim_sum(state: Nim) -> int:
    tmp = np.array([tuple(int(x) for x in f"{c:032b}") for c in state.rows])
    xor = tmp.sum(axis=0) % 2
    return int("".join(str(_) for _ in xor), base=2)


def analize(raw: Nim) -> dict:
    cooked = dict()
    cooked["possible_moves"] = dict()
    for ply in (Nimply(r, o) for r, c in enumerate(raw.rows) for o in range(1, c + 1)):
        tmp = deepcopy(raw)
        tmp.nimming(ply)
        cooked["possible_moves"][ply] = nim_sum(tmp)
    return cooked


def optimal(state: Nim) -> Nimply:
    analysis = analize(state)
    logging.debug(f"analysis:\n{pformat(analysis)}")
    spicy_moves = [ply for ply, ns in analysis["possible_moves"].items() if ns != 0]
    if not spicy_moves:
        spicy_moves = list(analysis["possible_moves"].keys())
    ply = random.choice(spicy_moves)
    return ply


def rule_based_nim_sum(state: Nim) -> Nimply:
    """A rule-based strategy using nim-sum to play Nim."""
    # Calculate the nim-sum for the current state
    current_nim_sum = nim_sum(state)

    # If the nim-sum is already zero, we cannot force a win, make a random move
    if current_nim_sum == 0:
        return pure_random(state)

    # Find a move that changes the current nim-sum to zero
    for r, num_objects in enumerate(state.rows):
        for remove in range(1, min(state._k, num_objects) + 1 if state._k else num_objects + 1):
            # Create a copy of the state and perform the move
            tmp_state = deepcopy(state)
            tmp_state.nimming(Nimply(r, remove))
            # Calculate the nim-sum after the move
            if nim_sum(tmp_state) == 0:
                return Nimply(r, remove)
    # If no such move is found, fall back to a random move
    return pure_random(state)


def fitness(state: Nim, strategy_func, opponent_func, num_trials=100):
    """Evaluate the performance of a strategy function against a given opponent."""
    wins = 0
    for _ in range(num_trials):
        temp_state = deepcopy(state)
        player = 0  # Start with the strategy we are evaluating
        while temp_state:
            if player == 0:
                ply = strategy_func(temp_state)
            else:
                ply = opponent_func(temp_state)
            temp_state.nimming(ply)
            if not temp_state:
                if player == 0:  # If the strategy we are evaluating wins
                    wins += 1
                break
            player = 1 - player
    return wins


def gaussian_mutation(strategy, mu=0, sigma=1):
    """Mutate the strategy parameters using Gaussian noise."""
    mutated_strategy = deepcopy(strategy)
    for key in mutated_strategy:
        mutated_strategy[key] += np.random.normal(mu, sigma)
    return mutated_strategy


def select_top_strategies(population, fitnesses, top_n):
    """Select the top N strategies based on their fitness scores."""
    indices = np.argsort(fitnesses)[-top_n:]
    return [population[i] for i in indices]


def evolve_strategy(state: Nim, population_size=10, generations=10, top_n=5, mutation_rate=0.1):
    """An evolutionary strategy to play Nim."""
    # Initialize a population of strategies with random parameters
    population = [{'param': random.uniform(-1, 1)} for _ in range(population_size)]

    for generation in range(generations):
        # Evaluate the fitness for each strategy in the population
        fitnesses = [fitness(state, lambda s, p=individual['param']: evolved_strategy(s, p), rule_based_nim_sum) for individual in population]

        # Select the top N strategies
        top_strategies = select_top_strategies(population, fitnesses, top_n)

        # Create a new population by mutating the top strategies
        new_population = [gaussian_mutation(strategy, sigma=mutation_rate) for strategy in top_strategies for _ in range(population_size // top_n)]

        # Replace the old population with the new population
        population = new_population + top_strategies[:population_size % top_n]

    # Return the best strategy from the final population
    best_fitness_index = np.argmax(fitnesses)
    return population[best_fitness_index]

def evolved_strategy(state: Nim, param):
    """A strategy function using an evolved parameter to influence move choice."""
    # Convert the parameter to a probability bias
    bias = (param + 1) / 2  # Now it's between 0 and 1

    current_nim_sum = nim_sum(state)

    if current_nim_sum == 0:
        # If the nim-sum is zero, make a move that tries to create a non-zero nim-sum
        for r, num_objects in enumerate(state.rows):
            for remove in range(1, min(state._k, num_objects) + 1 if state._k else num_objects + 1):
                tmp_state = deepcopy(state)
                tmp_state.nimming(Nimply(r, remove))
                if nim_sum(tmp_state) != 0:
                    return Nimply(r, remove)
        return pure_random(state)  # Fallback to random if no such move is found
    else:
        # If the nim-sum is non-zero, find a move that changes it to zero
        for r, num_objects in enumerate(state.rows):
            for remove in range(1, min(state._k, num_objects) + 1 if state._k else num_objects + 1):
                tmp_state = deepcopy(state)
                tmp_state.nimming(Nimply(r, remove))
                if nim_sum(tmp_state) == 0:
                    return Nimply(r, remove)
        return pure_random(state)  # Fallback to random if no such move is found

best_strategy_params = evolve_strategy(Nim(4, k=3), population_size=1, generations=10, top_n=5, mutation_rate=0.1)
evolved_strategy_function = lambda state: evolved_strategy(state, best_strategy_params['param'])
logging.getLogger().setLevel(logging.INFO)

strategy = (optimal, evolved_strategy_function)

nim = Nim(4)
logging.info(f"init : {nim}")
player = random.randint(0,1)
logging.info(f"Player {player} will start the game.")
logging.info(f"Initial state: {nim}")
while nim:
    ply = strategy[player](nim)
    logging.info(f"ply: player {player} plays {ply}")
    nim.nimming(ply)
    if not nim:  # Check if the game is over before switching players
        logging.info(f"status: {nim}")
        logging.info(f" Player {player} won!")
        break
    logging.info(f"status: {nim}")
    player = 1 - player
