from hnefatafl.core.gameState import GameState
from hnefatafl.agents.agent import RandomAgent, HumanAgent
from hnefatafl.core.gameTypes import Player
from hnefatafl.agents.mcts.MCTS_multiproccesor import MCTSAgent

def main():
    game = GameState.new_game()
    bots = {
        Player.black: RandomAgent(),
        Player.white: MCTSAgent(num_rounds=50, selection_strategy='value')
    }

    for i in range(5000):
        bot_move = bots[game.next_player].select_move(game)
        print(f"Bot {game.next_player} move: {bot_move}")
        game = game.apply_move(bot_move)
        print(game.board)
        if game.is_over():
            print(f"Game over! Winner: {game.winner}")
            print(game.board)
            print(f"Total moves: {i + 1}")
            break

if __name__ == '__main__':
    main()