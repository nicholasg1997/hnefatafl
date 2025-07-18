from hnefatafl.core.gameTypes import Player
from hnefatafl.agents.agent import Agent
import numpy as np
import random


class MCTSNode:
    def __init__(self, game_state, parent=None, move=None):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.win_counts = {Player.black: 0, Player.white: 0}
        self.num_rollouts = 0
        self.children = []
        self.untried_moves = None
        self._is_terminal = None

    def add_random_move(self):
        if self.untried_moves is None:
            self.untried_moves = self.game_state.get_legal_moves()
            random.shuffle(self.untried_moves)

        new_move = self.untried_moves.pop()
        new_state = self.game_state.apply_move(new_move)
        child_node = MCTSNode(new_state, parent=self, move=new_move)
        self.children.append(child_node)
        return child_node

    def record_win(self, winner):
        self.win_counts[winner] += 1
        self.num_rollouts += 1

    def can_add_child(self):
        if self.untried_moves is None:
            self.untried_moves = self.game_state.get_legal_moves()
            random.shuffle(self.untried_moves)
        return len(self.untried_moves) > 0

    def is_terminal(self):
        if self._is_terminal is None:
            self._is_terminal = self.game_state.is_over()
        return self._is_terminal

    def winning_frac(self, player):
        if self.num_rollouts == 0:
            return 0
        return float(self.win_counts[player]) / float(self.num_rollouts)


class MCTSAgent(Agent):
    def __init__(self, num_rounds=500, temperature=np.sqrt(2)):
        Agent.__init__(self)
        self.num_rounds = num_rounds
        self.temperature = temperature

    def select_move(self, game_state):
        root = MCTSNode(game_state)

        for _ in range(self.num_rounds):
            # Selection
            node = root
            while (not node.can_add_child()) and (not node.is_terminal()):
                node = self.select_child(node)

            # Expansion
            if node.can_add_child():
                node = node.add_random_move()

            # Simulation
            winner = self.simulate_fast(node.game_state)

            # Backpropagation
            while node is not None:
                node.record_win(winner)
                node = node.parent

        return self._select_best_move(root, game_state.next_player)

    def select_child(self, node):
        total_rollouts = sum(child.num_rollouts for child in node.children)
        log_rollouts = np.log(total_rollouts) if total_rollouts > 0 else 1

        best_value = -1.0
        best_child = None

        for child in node.children:
            win_ratio = child.winning_frac(node.game_state.next_player)
            if child.num_rollouts == 0:
                exploration_value = float('inf')
            else:
                exploration_value = np.sqrt(log_rollouts / child.num_rollouts)

            uct_score = win_ratio + self.temperature * exploration_value
            if uct_score > best_value:
                best_value = uct_score
                best_child = child

        return best_child

    def simulate_fast(self, game):
        """Optimized simulation without creating agent objects"""
        current_game_state = game
        while not current_game_state.is_over():
            legal_moves = current_game_state.get_legal_moves()
            if not legal_moves:
                return current_game_state.next_player.other
            random_move = random.choice(legal_moves)
            current_game_state = current_game_state.apply_move(random_move)
        return current_game_state.winner

    def _select_best_move(self, root, player):
        if not root.children:
            return None

        best_move = None
        best_value = -1.0
        for child in root.children:
            value = child.winning_frac(player)
            if value > best_value:
                best_value = value
                best_move = child.move
        return best_move

if __name__ == "__main__":
    from hnefatafl.core.gameState_fast import GameState

    # Run a sample game with MCTS agent
    game = GameState.new_game()
    mcts_agent = MCTSAgent(num_rounds=500)

    for i in range(5000):
        print(f"move {i + 1}")
        move = mcts_agent.select_move(game)
        print(f"MCTS Agent move: {move}")
        game = game.apply_move(move)
        print(game.board)
        if game.is_over():
            print(f"Game over! Winner: {game.winner}")
            print(game.board)
            print(f"Total moves: {i + 1}")
            break
