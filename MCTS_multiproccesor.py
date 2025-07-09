from gameTypes import Player
import agent
import numpy as np
from multiprocessing import Pool, cpu_count
from MCTS import MCTSNode
import random


def simulate(game):
    """
    Runs a single random game simulation from a given game state.
    """
    # To avoid creating new agent objects in every simulation, we can simplify.
    current_game_state = game
    while not current_game_state.is_over():
        legal_moves = current_game_state.get_legal_moves()
        if not legal_moves:
            # If the current player has no moves, they lose.
            return current_game_state.next_player.other

        random_move = random.choice(legal_moves)
        current_game_state = current_game_state.apply_move(random_move)
    return current_game_state.winner


class MCTSAgent(agent.Agent):
    def __init__(self, num_rounds=200, temperature=np.sqrt(2), rollouts_per_leaf=None, selection_strategy='visits'):
        """
        Initializes the MCTS agent.

        :param num_rounds: The number of selection/expansion/simulation cycles to run.
        :param temperature: The exploration constant (UCT).
        :param rollouts_per_leaf: The number of parallel simulations to run for each selected leaf. Defaults to cpu_count().
        :param selection_strategy: 'visits' or 'value' for final move selection.
        """
        agent.Agent.__init__(self)
        self.num_rounds = num_rounds
        self.temperature = temperature
        self.rollouts_per_leaf = rollouts_per_leaf if rollouts_per_leaf is not None else cpu_count()
        self.selection_strategy = selection_strategy
        print(
            f"MCTS Agent initialized: {self.num_rounds} rounds, {self.rollouts_per_leaf} rollouts/leaf, '{self.selection_strategy}' strategy.")

    def select_move(self, game_state):
        root = MCTSNode(game_state)

        # The main loop now iterates for num_rounds, performing the MCTS cycle each time.
        for i in range(self.num_rounds):
            if i > 0 and i % 50 == 0:
                print(f"  ... MCTS round {i}/{self.num_rounds}")

            # 1. Selection: Traverse the tree to find a promising node.
            node = root
            while (not node.can_add_child()) and (not node.is_terminal()):
                node = self.select_child(node)

            # 2. Expansion: If the node has untried moves, expand it.
            if node.can_add_child():
                node = node.add_random_move()

            # 3. Simulation (in parallel) & 4. Backpropagation
            if not node.is_terminal():
                # Prepare the tasks for the parallel pool.
                simulation_tasks = [node.game_state] * self.rollouts_per_leaf

                # Use a Pool to run the simulations in parallel.
                with Pool(processes=self.rollouts_per_leaf) as pool:
                    winners = pool.map(simulate, simulation_tasks)

                # Backpropagate all results from this leaf's simulations.
                temp_node = node
                while temp_node is not None:
                    for winner in winners:
                        temp_node.record_win(winner)
                    temp_node = temp_node.parent

        print(f"  ... MCTS round {self.num_rounds}/{self.num_rounds} complete.")
        # After all rounds are done, select the best move from the root's children.
        best_move = self._select_best_move(root, game_state.next_player)
        return best_move

    def select_child(self, node):
        total_rollouts = sum(child.num_rollouts for child in node.children)
        log_rollouts = np.log(total_rollouts) if total_rollouts > 1 else 1.0

        best_uct_score = -1.0
        best_child = None

        for child in node.children:
            win_ratio = child.winning_frac(node.game_state.next_player)

            if child.num_rollouts == 0:
                exploration_value = float('inf')
            else:
                exploration_value = np.sqrt(log_rollouts / child.num_rollouts)

            uct_score = win_ratio + self.temperature * exploration_value

            if uct_score > best_uct_score:
                best_uct_score = uct_score
                best_child = child

        return best_child if best_child is not None else random.choice(node.children)

    def _select_best_move(self, root, player):
        if not root.children:
            print("No children found. Cannot select a move.")
            return None

        best_move = None
        if self.selection_strategy == 'visits':
            best_value = -1
            for child in root.children:
                if child.num_rollouts > best_value:
                    best_value = child.num_rollouts
                    best_move = child.move
            print(f"Selected move {best_move} with {best_value} visits.")
            return best_move

        elif self.selection_strategy == 'value':
            best_value = -1.0
            for child in root.children:
                value = child.winning_frac(player)
                if value > best_value:
                    best_value = value
                    best_move = child.move
            print(f"Selected move {best_move} with a win rate of {best_value:.2%}.")
            return best_move
        else:
            raise ValueError("Invalid selection strategy. Use 'visits' or 'value'.")


if __name__ == "__main__":
    from gameState import GameState
    import time

    # A good starting point for testing is a lower number of rounds.
    # Total simulations per move = num_rounds * rollouts_per_leaf
    # e.g., 100 rounds * 8 cores = 800 simulations per move.
    mcts_agent = MCTSAgent(num_rounds=100, selection_strategy='visits')
    game = GameState.new_game()

    for i in range(5000):
        print(f"\nMove {i + 1} (Player: {game.next_player.name})")
        start_time = time.time()

        move = mcts_agent.select_move(game)

        end_time = time.time()
        print(f"Move selection took {end_time - start_time:.2f} seconds.")

        if move is None:
            print("Agent returned no move. Game over.")
            break

        game = game.apply_move(move)
        print(game.board)

        if game.is_over():
            print(f"\nGame over! Winner: {game.winner}")
            print(f"Total moves: {i + 1}")
            break
