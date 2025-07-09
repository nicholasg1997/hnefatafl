from hnefatafl.core.gameState import GameState
from hnefatafl.agents.agent import HumanAgent
from hnefatafl.agents.mcts.MCTS_multiproccesor import MCTSAgent
from hnefatafl.core.gameTypes import Player

def main():
    game = GameState.new_game()
    agents = {
        Player.black: HumanAgent(),  # Human plays as black
        Player.white: MCTSAgent(num_rounds=50, selection_strategy='value')  # AI plays as white
    }

    print("Welcome to Hnefatafl! You are playing as Black (attackers).")
    print("Enter moves in the format 'A1 B2' to move from A1 to B2.")
    print("The white player (defenders) is controlled by the AI.")
    print("\nInitial board:")
    print(game.board)

    for i in range(5000):
        print(f"\nMove {i + 1}, {game.next_player.name} to play")
        
        try:
            move = agents[game.next_player].select_move(game)
            print(f"{game.next_player.name} plays: {move}")
            game = game.apply_move(move)
            print(game.board)
            
            if game.is_over():
                print(f"\nGame over! Winner: {game.winner}")
                print("Final Board:")
                print(game.board)
                print(f"Total moves: {i + 1}")
                break
        except ValueError as e:
            print(f"Error: {e}")
            if game.next_player == Player.black:  # Only retry for human player
                print("Please try again.")
            else:
                print("AI made an invalid move. Game ending.")
                break

if __name__ == '__main__':
    main()