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
        return [(random.randint(0, 4), random.randint(0, 4), random.choice(list(Move))) for _ in range(5)]

    def mutate(self, mutation_rate):
        for i in range(len(self.move_sequence)):
            if random.random() < mutation_rate:
                new_pos = (random.randint(0, 4), random.randint(0, 4))
                new_direction = random.choice(list(Move))
                self.move_sequence[i] = (new_pos[0], new_pos[1], new_direction)
    @staticmethod
    def crossover(parent1, parent2):
        child_move_sequence = []
        for i in range(len(parent1.move_sequence)):
            if random.random() < 0.5:
                child_move_sequence.append(parent1.move_sequence[i])
            else:
                child_move_sequence.append(parent2.move_sequence[i])
        return Individual(move_sequence=child_move_sequence)
class ESAgent(Player):
    def __init__(self, player_id, generations: int = 200, population_size: int = 100):
        super().__init__()
        self.player_id = player_id
        self.generations = generations
        self.population_size = population_size
        self.population = [Individual() for _ in range(self.population_size)]

    def make_move(self, game):
        for generation in range(self.generations):
            for individual in self.population:
                mutation_rate = self.adaptive_mutation_rate(generation)
                individual.mutate(mutation_rate)

            scores = [self.evaluate_move_sequence(ind.move_sequence, game) for ind in self.population]
            best_index = scores.index(max(scores))
            best_individual = self.population[best_index]

            if generation < self.generations - 1:
                next_generation = [Individual.crossover(random.choice(self.population), random.choice(self.population)) for _ in range(self.population_size)]
                self.population = next_generation
            else:
                move = best_individual.move_sequence[0]
                game.print()
                return move[0:2], move[2]

    def adaptive_mutation_rate(self, generation):
        return 0.1 - (generation / (2 * self.generations)) * 0.09

    def evaluate_move_sequence(self, move_sequence, game):
        simulated_game = SimulatedGame(game)
        for move in move_sequence:
            from_pos, move_dir = move[0:2], move[2]
            simulated_game.simulate_move(from_pos, move_dir, self.player_id)
        score = self.calculate_score(simulated_game)
        return score

    def calculate_score(self, simulated_game):
        player_score = 0
        opponent_score = 0
        board = simulated_game.get_board()

        # Enhanced evaluation logic here, considering strategic positions
        lines = self.get_all_lines(board)
        center_control = board[2, 2] == self.player_id
        player_score += 50 if center_control else 0

        for line in lines:
            if np.all(line == self.player_id):
                player_score += 100
            elif np.all(line == 1 - self.player_id):
                opponent_score += 100
            else:
                player_score += np.count_nonzero(line == self.player_id) * 10
                opponent_score += np.count_nonzero(line == 1 - self.player_id) * 10

        return player_score - opponent_score

    def get_all_lines(self, board):
        lines = []
        # Horizontal and vertical
        for i in range(5):
            lines.append(board[i, :])  # Horizontal
            lines.append(board[:, i])  # Vertical
        # Diagonals
        lines.append(np.diag(board))
        lines.append(np.diag(np.fliplr(board)))
        return lines

    def tournament_selection(self, scores, k=3):
        tournament = random.sample(list(enumerate(self.population)), k)
        tournament_winner = max(tournament, key=lambda x: scores[x[0]])
        return tournament_winner[1]

class MinMaxPlayer(Player):
    def __init__(self, player_id: int, max_depth=3) -> None:
        super().__init__()
        self.max_depth = max_depth
        self.player_id = player_id
        self.first_move_done = False

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        if not self.first_move_done:
            from_pos = (random.randint(0, 4), random.randint(0, 4))
            move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
            self.first_move_done = True
            game.print()
            return from_pos, move
        else:
            _, move = self.minmax_alpha_beta(game, 0, float('-inf'), float('inf'), True)
            game.print()
            return move

    def minmax_alpha_beta(self, game, depth, alpha, beta, is_maximizing_player):
        winner = game.check_winner()
        if winner != -1 or depth == self.max_depth:
            return self.evaluate_game(game, winner), None
        depth += 1
        best_move = None

        if is_maximizing_player:
            max_eval = float('-inf')
            for move in self.get_possible_moves(game, self.player_id):
                eval = self.evaluate_maximizing_player(game, move, depth, alpha, beta)
                if eval > max_eval:
                    max_eval, best_move = eval, move
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
        else:
            min_eval = float('inf')
            for move in self.get_possible_moves(game, 1 - self.player_id):
                eval = self.evaluate_minimizing_player(game, move, depth, alpha, beta)
                if eval < min_eval:
                    min_eval, best_move = eval, move
                beta = min(beta, eval)
                if alpha >= beta:
                    break
        return (max_eval if is_maximizing_player else min_eval), best_move

    def evaluate_game(self, game, winner):
        if winner == self.player_id:
            return float('inf')
        elif winner != -1:
            return float('-inf')
        else:
            num_self = np.count_nonzero(game.get_board() == self.player_id)
            num_opponent = np.count_nonzero(game.get_board() == 1 - self.player_id)
            return (num_self - num_opponent) / 100

    def evaluate_maximizing_player(self, game, move, depth, alpha, beta):
        new_game = self.get_new_board(game, move, self.player_id)
        eval, _ = self.minmax_alpha_beta(new_game, depth, alpha, beta, False)
        return eval

    def evaluate_minimizing_player(self, game, move, depth, alpha, beta):
        new_game = self.get_new_board(game, move, 1 - self.player_id)
        eval, _ = self.minmax_alpha_beta(new_game, depth, alpha, beta, True)
        return eval

    def get_new_board(self, game, possible_move, player):
        new_game = deepcopy(game)
        pos, move = possible_move
        if not new_game._Game__move(pos, move, player):
            print(f'ERROR: Invalid move {pos}, {move} for player {player}')
            game.print()
        return new_game

    def get_possible_moves(self, game, player):
        possible_pos = [(i, j) for i in range(5) for j in range(5)
                        if (i in [0, 4] or j in [0, 4]) and (game._board[j, i] in [-1, player])]
        possible_moves = [(pos, mov) for pos in possible_pos for mov in [Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT]]
        return [(pos, mov) for pos, mov in possible_moves if self.acceptable_move(pos, mov)]

    def acceptable_move(self, pos, mov):
        return not ((pos[1] == 0 and mov == Move.TOP) or (pos[1] == 4 and mov == Move.BOTTOM)
                    or (pos[0] == 0 and mov == Move.LEFT) or (pos[0] == 4 and mov == Move.RIGHT))


if __name__ == '__main__':
    Player1_w = 0
    Player2_w = 0
    Draw = 0
    Num = 20
    for i in range(Num):
        g = Game()
        player1 = MinMaxPlayer(0)
        player2 = ESAgent(1)
        print(f"Game {i}")
        winner = g.play(player1, player2)
        print(f"Winner: Player {winner}")
        if winner == 0:
            Player1_w += 1
        elif winner == 1:
            Player2_w += 1
        else:
            Draw += 1
    print(f"Winrate: {player1.__class__.__name__} ={(Player1_w / Num)*100} % - {player2.__class__.__name__} = {(Player2_w / Num)*100}% - Draw = {(Draw / Num)*100}%")