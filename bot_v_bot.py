from gameState import GameState
from agent import RandomAgent, HumanAgent
from gameTypes import Player

def main():
    game = GameState.new_game()
    bots = {
        Player.black: RandomAgent(),
        Player.white: HumanAgent()
    }

    for i in range(5000):
        bot_move = bots[game.next_player].select_move(game)
        print(f"Bot {game.next_player} move: {bot_move}")
        game = game.apply_move(bot_move)
        if game.is_over():
            print(f"Game over! Winner: {game.winner}")
            print(game.board)
            print(f"Total moves: {i + 1}")
            break

if __name__ == '__main__':
    main()
