from hnefatafl.encoders.advanced_encoder import SevenPlaneEncoder
from hnefatafl.zero.zeroagent import ZeroAgent
from hnefatafl.zero.network import DualNetwork
from hnefatafl.zero.experiencecollector import ZeroExperienceCollector, combine_experience
from hnefatafl.core.gameState import GameState
from hnefatafl.core.gameTypes import Player
from hnefatafl.utils.nnTrainingUtils import simulate_game_simple as simulate_game
from tqdm import tqdm

import torch

def main(learning_rate=0.01, batch_size=16, num_generations = 10,
         num_self_play_games=2, num_training_epochs=1, mcts_rounds=25, model_save_freq=10):
    board_size = 11
    num_generations = num_generations
    learning_rate = learning_rate
    batch_size = batch_size
    num_self_play_games = num_self_play_games
    num_training_epochs = num_training_epochs
    mcts_rounds = mcts_rounds
    model_save_freq = model_save_freq

    encoder = SevenPlaneEncoder(board_size)
    model = DualNetwork(encoder, learning_rate=learning_rate)

    for generation in range(num_generations):
        print(f"Starting generation {generation + 1}/{num_generations}")
        collectors = []
        for i in tqdm(range(num_self_play_games)):
            print(f"Starting game {i + 1}/{num_self_play_games}")

            black_agent = ZeroAgent(model, encoder, rounds_per_move=mcts_rounds)
            white_agent = ZeroAgent(model, encoder, rounds_per_move=mcts_rounds)

            c1 = ZeroExperienceCollector()
            c2 = ZeroExperienceCollector()
            black_agent.set_collector(c1)
            white_agent.set_collector(c2)

            c1.begin_episode()
            c2.begin_episode()

            winner = simulate_game(black_agent, white_agent,max_moves=5, verbose=False)

            if winner == Player.black:
                print("Black wins!")
                c1.complete_episode(1.0)
                c2.complete_episode(0.0)
            elif winner == Player.white:
                print("White wins!")
                c1.complete_episode(0.0)
                c2.complete_episode(1.0)
            else:
                print("Game ended in a draw.")
                c1.complete_episode(-0.5)
                c2.complete_episode(-0.5)

            collectors.append(c1)
            collectors.append(c2)

        experience = combine_experience(collectors)
        print("Training model...")
        black_agent.train(experience, batch_size, num_training_epochs)

        if (generation + 1) % model_save_freq == 0:
            print(f"Saving model after generation {generation + 1}")
            torch.save(model.state_dict(), f'models/model_gen_{generation+1}.pth')
            print("Model saved.")

    torch.save(model.state_dict(), 'models/model_final.pth')
    print("Training complete. Final model saved.")

if __name__ == "__main__":
    main(num_generations=100, num_self_play_games=200, num_training_epochs=10, mcts_rounds=500, batch_size=128, learning_rate=0.001)
