from hnefatafl.encoders.advanced_encoder import SevenPlaneEncoder
from hnefatafl.zero.zeroagent import ZeroAgent
from hnefatafl.zero.network import DualNetwork
from hnefatafl.zero.experiencecollector import ZeroExperienceCollector, combine_experience
from hnefatafl.core.gameState import GameState
from hnefatafl.core.gameTypes import Player

import torch

def main():
    board_size = 11
    learning_rate = 0.01
    batch_size = 128
    num_self_play_games = 10
    num_training_epochs = 5
    mcts_rounds = 100

    encoder = SevenPlaneEncoder(board_size)
    model = DualNetwork(encoder, learning_rate=learning_rate)
    collectors = []
    for i in range(num_self_play_games):
        print(f"Starting game {i + 1}/{num_self_play_games}")

        black_agent = ZeroAgent(model, encoder, rounds_per_move=mcts_rounds)
        white_agent = ZeroAgent(model, encoder, rounds_per_move=mcts_rounds)

        c1 = ZeroExperienceCollector()
        c2 = ZeroExperienceCollector()
        black_agent.set_collector(c1)
        white_agent.set_collector(c2)

        game = GameState.new_game(board_size)
        c1.begin_episode()
        c2.begin_episode()

        while not game.is_over():
            if game.next_player == Player.black:
                move = black_agent.select_move(game)
            else:
                move = white_agent.select_move(game)

            game = game.apply_move(move)

        if game.winner == Player.black:
            c1.end_episode(1.0)
            c2.end_episode(0.0)
        elif game.winner == Player.white:
            c1.end_episode(0.0)
            c2.end_episode(1.0)
        else:
            c1.end_episode(0.0)
            c2.end_episode(0.0)

        collectors.append(c1)
        collectors.append(c2)

    experience = combine_experience(collectors)

    black_agent.train(experience, batch_size, num_training_epochs)

    torch.save(model.state_dict(), 'model.pth')
    print("Model saved.")

if __name__ == "__main__":
    main()
