from game import SimulatedGame,Game, Move, Player
import random
import numpy as np
from copy import deepcopy
class RandomPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        game.print()
        return from_pos, move
class Individual:
    def __init__(self, move_sequence=None):
        self.move_sequence = move_sequence or self.generate_random_sequence()

    def generate_random_sequence(self):
        # Generate a random sequence of moves
        return [
            (random.randint(0, 4), random.randint(0, 4), random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT]))
            for _ in range(5)]

    def mutate(self, mutation_rate):
        # Mutate the individual's move sequence based on the mutation rate
        for i in range(len(self.move_sequence)):
            if random.random() < mutation_rate:
                self.move_sequence[i] = (random.randint(0, 4), random.randint(0, 4), random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT]))

    @staticmethod
    def crossover(parent1, parent2):
        # Create offspring with a combination of moves from both parents
        child_move_sequence = []
        for i in range(len(parent1.move_sequence)):
            if random.random() < 0.5:
                child_move_sequence.append(parent1.move_sequence[i])
            else:
                child_move_sequence.append(parent2.move_sequence[i])
        return Individual(move_sequence=child_move_sequence)

class ESAgent(Player):
    def __init__(self, player_id: int, generations: int = 100, population_size: int = 50) -> None:
        super().__init__()
        self.player_id = player_id
        self.generations = generations
        self.population_size = population_size

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        population = [Individual() for _ in range(self.population_size)]

        for generation in range(self.generations):
            # Apply adaptive mutation rate for this generation
            mutation_rate = self.adaptive_mutation_rate(generation)

            # Mutate the population
            for individual in population:
                individual.mutate(mutation_rate)

            # Evaluate the population
            scores = [self.evaluate_move_sequence(ind.move_sequence, game) for ind in population]
            elite_indices = self.select_elite_indices(scores)
            best_individual = population[elite_indices[0]]

            # Introduce diversity into the population every few generations
            if generation % 10 == 0:
                population = self.introduce_diversity(population)

            # Crossover and selection for the next generation
            next_generation = []
            while len(next_generation) < self.population_size:
                # Randomly select two parents for crossover
                parent1, parent2 = random.sample(population, 2)
                # Create offspring
                offspring = Individual.crossover(parent1, parent2)
                next_generation.append(offspring)

            # Select the best-performing individuals for the next generation
            next_generation_scores = [self.evaluate_move_sequence(ind.move_sequence, game) for ind in next_generation]
            best_indices = self.select_elite_indices(next_generation_scores)
            population = [next_generation[i] for i in best_indices]

            # Make the move specified by the best individual (if it's the last generation)
            if generation == self.generations - 1:
                move = best_individual.move_sequence[0]
                return move[0:2], move[2]

    def evaluate_move_sequence(self, move_sequence, game):
        # Simulate the game with the given move sequence and evaluate the final state
        simulated_game = SimulatedGame(game)
        simulated_game.current_player_idx = self.player_id

        for move in move_sequence:
            from_pos, slide = move[0:2], move[2]
            simulated_game._Game__move(from_pos, slide, self.player_id)

        # Evaluate the final game state
        return self.evaluate(simulated_game)

    def evaluate(self, game: 'Game'):
        player_positions = [(x, y) for x in range(5) for y in range(5) if game.get_board()[x, y] == self.player_id]

        # Check for winning positions horizontally, vertically, and diagonally
        score = 0
        # Enhanced evaluation for strategic decision-making
        for position in player_positions:
            x, y = position
            # Weighted scoring for different line types
            horizontal_score = sum(game.get_board()[x, i] == self.player_id for i in range(5))
            vertical_score = sum(game.get_board()[i, y] == self.player_id for i in range(5))
            diagonal_score = sum(game.get_board()[i, i] == self.player_id for i in range(5)) if x == y else 0
            anti_diagonal_score = sum(
                game.get_board()[i, 4 - i] == self.player_id for i in range(5)) if x + y == 4 else 0

            # Assign weights to different line types based on strategic value
            score += (horizontal_score * 2 + vertical_score * 2 + diagonal_score * 3 + anti_diagonal_score * 3)

            # Consider opponent's potential winning positions (defensive strategy)
            opponent_id = 1 if self.player_id == 0 else 0
            opponent_score = sum(game.get_board()[x, i] == opponent_id for i in range(5)) * -2
            score += opponent_score

        return score

    def select_elite_indices(self, scores):
        # Select the indices of the top-performing individuals
        return sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)[:min(5, len(scores))]

    def introduce_diversity(self, population):
        # Introduce new random individuals to the population
        diversity_count = int(self.population_size * 0.1)  # 10% of the population
        for _ in range(diversity_count):
            population[random.randint(0, len(population) - 1)] = Individual()
        return population

    def adaptive_mutation_rate(self, generation):
        # Adjust the mutation rate based on the generation number
        initial_rate = 0.1  # 10% mutation rate initially
        final_rate = 0.01   # 1% mutation rate in the final generation
        return initial_rate - (initial_rate - final_rate) * (generation / self.generations)


class MinMaxPlayer(Player):
    def __init__(self, max_depth=4) -> None:
        super().__init__()
        self.max_depth = max_depth

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        _, move = self.minmax_alpha_beta(game, 0, -float('inf'), float('inf'), True)
        return move

    def minmax_alpha_beta(self, game, depth, alpha, beta, is_maximizing_player):
        winner = game.check_winner()
        if winner != -1 or depth == self.max_depth:
            return self.evaluate_game(game, winner), None

        depth += 1
        best_move = None

        if is_maximizing_player:
            max_eval = float('-inf')
            for move in self.get_possible_moves(game, 0):
                eval = self.evaluate_maximizing_player(game, move, depth, alpha, beta)
                if eval > max_eval:
                    max_eval, best_move = eval, move
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
        else:
            min_eval = float('inf')
            for move in self.get_possible_moves(game, 1):
                eval = self.evaluate_minimizing_player(game, move, depth, alpha, beta)
                if eval < min_eval:
                    min_eval, best_move = eval, move
                beta = min(beta, eval)
                if alpha >= beta:
                    break

        return (max_eval if is_maximizing_player else min_eval), best_move

    def evaluate_game(self, game, winner):
        if winner == 0:
            return 1
        elif winner == 1:
            return -1
        else:
            num_zeros = np.count_nonzero(game._board == 0)
            num_ones = np.count_nonzero(game._board == 1)
            return (num_zeros - num_ones) / 100

    def evaluate_maximizing_player(self, game, move, depth, alpha, beta):
        new_game = self.get_new_board(game, move, 0)
        eval, _ = self.minmax_alpha_beta(new_game, depth, alpha, beta, False)
        return eval

    def evaluate_minimizing_player(self, game, move, depth, alpha, beta):
        new_game = self.get_new_board(game, move, 1)
        eval, _ = self.minmax_alpha_beta(new_game, depth, alpha, beta, True)
        return eval

    def get_new_board(self, game, possible_move, player):
        # We do a deepcopy of the game object because in MinMax we change the game, not only the board attribute
        new_game = deepcopy(game)
        pos, move = possible_move
        if not new_game._Game__move(pos, move, player):
            print(f'ERROR: Invalid move {pos}, {move} for player {player}')
            game.print()
        return new_game

    def acceptable_move(self, pos, mov):
        return not ((pos[1] == 0 and mov == Move.TOP)
                    or (pos[1] == 4 and mov == Move.BOTTOM)
                    or (pos[0] == 0 and mov == Move.LEFT)
                    or (pos[0] == 4 and mov == Move.RIGHT))

    def get_possible_moves(self, game, player):
        possible_pos = [(i, j) for i in range(5) for j in range(5)
                        if (i in [0, 4] or j in [0, 4]) and (game._board[j, i] in [-1, player])]

        possible_moves = [(pos, mov) for pos in possible_pos for mov in [Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT]]
        return [(pos, mov) for pos, mov in possible_moves if self.acceptable_move(pos, mov)]

if __name__ == '__main__':
    g = Game()
    g.print()
    # Change the players based on your requirements
    player1 = ESAgent(0)
    player2 = RandomPlayer()

    winner = g.play(player1, player2)
    g.print()
    print(f"Winner: Player {winner}")

