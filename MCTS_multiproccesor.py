from gameTypes import Player
import random
import agent
import numpy as np
from multiprocessing import Pool, cpu_count

class MCTSNode:
    def __init__(self, game_state, parent=None, move=None):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.win_counts = {Player.black: 0, Player.white: 0}
        self.num_rollouts = 0
        self.children = []
        self.untried_moves = game_state.get_legal_moves()
        random.shuffle(self.untried_moves)

    def add_random_move(self):
        new_move = self.untried_moves.pop()
        new_state = self.game_state.apply_move(new_move)
        child_node = MCTSNode(new_state, parent=self, move=new_move)
        self.children.append(child_node)
        return child_node

    def record_win(self, winner):
        self.win_counts[winner] += 1
        self.num_rollouts += 1

    def can_add_child(self):
        return len(self.untried_moves) > 0

    def is_terminal(self):
        return self.game_state.is_over()

    def winning_frac(self, player):
        if self.num_rollouts == 0:
            return 0
        return float(self.win_counts[player]) / float(self.num_rollouts)


def simulate(game):
    bots = {
        Player.black: agent.RandomAgent(),
        Player.white: agent.RandomAgent()
    }
    while not game.is_over():
        move = bots[game.next_player].select_move(game)
        game = game.apply_move(move)
    return game.winner


class MCTSAgent(agent.Agent):
    def __init__(self, num_rounds=1000, temperature=np.sqrt(2), num_processes=None, selection_strategy='value'):
        agent.Agent.__init__(self)
        self.num_rounds = num_rounds
        self.temperature = temperature
        self.num_processes = num_processes if num_processes is not None else cpu_count()
        self.selection_strategy = selection_strategy

    def select_move(self, game_state):
        root = MCTSNode(game_state)

        simulations = []
        for _ in range(self.num_rounds):
            node = root
            while (not node.can_add_child()) and (not node.is_terminal()):
                node = self.select_child(node)

            if node.can_add_child():
                node = node.add_random_move()

            simulations.append((node, node.game_state))

        # Simulate all games in parallel
        with Pool(processes=self.num_processes) as pool:
            results = pool.map(simulate, [sim[1] for sim in simulations])

        # Backpropagate results
        for (node, _), winner in zip(simulations, results):
            current_node = node
            while current_node is not None:
                current_node.record_win(winner)
                current_node = current_node.parent

        best_move = self._select_best_move(root, game_state.next_player)
        return best_move

    def select_child(self, node):
        total_rollouts = sum(child.num_rollouts for child in node.children)
        log_rollouts = np.log(total_rollouts) if total_rollouts > 0 else 1

        best_value = -1.0
        best_child = None

        for child in node.children:
            win_ratio = child.winning_frac(node.game_state.next_player)
            exploration_value = np.sqrt(log_rollouts / child.num_rollouts) if child.num_rollouts > 0 else float('inf')
            uct_score = win_ratio + self.temperature * exploration_value
            if uct_score > best_value:
                best_value = uct_score
                best_child = child
        return best_child

    def _select_best_move(self, root, player):
        if not root.children:
            print("No children found. Cannot select a move.")
            return None

        if self.selection_strategy == 'visits':
            best_value = -1
            best_move = None
            for child in root.children:
                if child.num_rollouts > best_value:
                    best_value = child.num_rollouts
                    best_move = child.move
            print(f"Best move: {best_move} with {best_value} visits.")
            return best_move

        elif self.selection_strategy == 'value':
            best_value = -1.0
            best_move = None
            for child in root.children:
                value = child.winning_frac(player)
                if value > best_value:
                    best_value = value
                    best_move = child.move
            print(f"Best move: {best_move} with value {best_value}.")
            return best_move

        else:
            raise ValueError("Invalid selection strategy. Use 'visits' or 'value'.")


if __name__ == "__main__":
    from gameState import GameState

    game = GameState.new_game()
    mcts_agent = MCTSAgent(num_rounds=1_000, selection_strategy='value')

    for i in range(5000):
        print(f"Move {i + 1}")
        move = mcts_agent.select_move(game)
        print(f"MCTS Agent move: {move}")
        game = game.apply_move(move)
        print(game.board)
        if game.is_over():
            print(f"Game over! Winner: {game.winner}")
            print(game.board)
            print(f"Total moves: {i + 1}")
            break
