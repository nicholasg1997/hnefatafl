from hnefatafl.core.gameTypes import Player
import random
from hnefatafl.agents.agent import Agent, RandomAgent
import numpy as np

class MCTSNode(object):
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
        """
        Records a win for the specified player.

        :param winner: The player who won the game.
        """
        self.win_counts[winner] += 1
        self.num_rollouts += 1

    def can_add_child(self):
        """
        Checks if there are any untried moves left to add as children.

        :return: True if there are untried moves, False otherwise.
        """
        return len(self.untried_moves) > 0

    def is_terminal(self):
        """
        Checks if the current game state is terminal (i.e., the game is over).

        :return: True if the game is over, False otherwise.
        """
        return self.game_state.is_over()

    def winning_frac(self, player):
        """
        Calculates the winning fraction for the specified player.

        :param player: The player for whom to calculate the winning fraction.
        :return: The winning fraction for the specified player.
        """
        if self.num_rollouts == 0:
            return 0
        return float(self.win_counts[player]) / float(self.num_rollouts)

class MCTSAgent(Agent):
    def __init__(self, num_rounds=1000, temperature=np.sqrt(2)):
        """
        Initializes the MCTS agent with the specified number of rounds and temperature.

        :param num_rounds: The number of rounds to simulate in the MCTS.
        :param temperature: The temperature parameter for exploration.
        """
        Agent.__init__(self)
        self.num_rounds = num_rounds
        self.temperature = temperature

    def select_move(self, game_state):
        root = MCTSNode(game_state)
        for _ in range(self.num_rounds):
            node = root
            while (not node.can_add_child()) and (not node.is_terminal()):
                node = self.select_child(node)

            if node.can_add_child():
                node = node.add_random_move()

            winner = self.simulate(node.game_state)

            while node is not None:
                node.record_win(winner)
                node = node.parent

        best_move = None
        best_value = -1.0
        for child in root.children:
            value = child.winning_frac(game_state.next_player)
            if value > best_value:
                best_value = value
                best_move = child.move
        print(f"Best move: {best_move} with value {best_value}")
        return best_move

    def select_child(self, node):
        total_rollouts = sum(child.num_rollouts for child in node.children)
        log_rollouts = np.log(total_rollouts) if total_rollouts > 0 else 1

        best_value = -1.0
        best_child = []

        for child in node.children:
            win_ratio = child.winning_frac(node.game_state.next_player)
            exploration_value = np.sqrt(log_rollouts / child.num_rollouts) if child.num_rollouts > 0 else float('inf')
            uct_score = win_ratio + self.temperature * exploration_value
            if uct_score > best_value:
                best_value = uct_score
                best_child = child
        return best_child

    @staticmethod
    def simulate(game):
        bots = {
            Player.black: RandomAgent(),
            Player.white: RandomAgent()
        }
        while not game.is_over():
            move = bots[game.next_player].select_move(game)
            game = game.apply_move(move)
        return game.winner

if __name__ == "__main__":
    from hnefatafl.core.gameState import GameState

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
