from gameState import GameState
from MCTS_multiproccesor import MCTSAgent


def run_game():
    """
    Sets up and runs a game between two agents.
    """
    game = GameState.new_game()
    mcts_agent = MCTSAgent(num_rounds=50, selection_strategy='value')

    for i in range(5000):
        print(f"Move {i + 1}")
        move = mcts_agent.select_move(game)

        if move is None:
            print("Agent returned no move. Game over.")
            break

        print(f"MCTS Agent as {game.next_player.name} plays: {move}")
        game = game.apply_move(move)
        print(game.board)

        if game.is_over():
            print(f"\nGame over! Winner: {game.winner}")
            print("Final Board:")
            print(game.board)
            print(f"Total moves: {i + 1}")
            break

if __name__ == "__main__":
    run_game()