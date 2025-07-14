import multiprocessing
from functools import partial

from hnefatafl.encoders.advanced_encoder import SevenPlaneEncoder
from hnefatafl.zero.zeroagent_fast import ZeroAgent
from hnefatafl.zero.network import DualNetwork
from hnefatafl.zero.experiencecollector import ZeroExperienceCollector, combine_experience
from hnefatafl.core.gameState import GameState
from hnefatafl.core.gameTypes import Player
from hnefatafl.utils.nnTrainingUtils import simulate_game
from tqdm import tqdm

import torch

def run_self_play_game(mcts_rounds, model_state_dict, learning_rate, _):
    encoder = SevenPlaneEncoder(11)
    model = DualNetwork(encoder, learning_rate=learning_rate)
    model.load_state_dict(model_state_dict)
    model.eval()

    black_agent = ZeroAgent(model, encoder, rounds_per_move=mcts_rounds)
    white_agent = ZeroAgent(model, encoder, rounds_per_move=mcts_rounds)

    c1 = ZeroExperienceCollector()
    c2 = ZeroExperienceCollector()
    black_agent.set_collector(c1)
    white_agent.set_collector(c2)

    c1.begin_episode()
    c2.begin_episode()

    winner = simulate_game(black_agent, white_agent)

    if winner == Player.black:
        c1.complete_episode(1.0)
        c2.complete_episode(-1.0)
    elif winner == Player.white:
        c1.complete_episode(-1.0)
        c2.complete_episode(1.0)
    else:
        c1.complete_episode(0.0)
        c2.complete_episode(0.0)
    return c1, c2


def main(learning_rate=0.001, batch_size=16, num_generations = 10,
         num_self_play_games=2, num_training_epochs=1, mcts_rounds=25, model_save_freq=10,
         num_workers=None):
    board_size = 11

    encoder = SevenPlaneEncoder(board_size)
    model = DualNetwork(encoder, learning_rate=learning_rate)

    if num_workers is None:
        num_workers = torch.multiprocessing.cpu_count()
    print(f"Using {num_workers} workers.")

    for generation in range(num_generations):
        print(f"Starting generation {generation + 1}/{num_generations}")

        model.eval()
        model_state_dict = model.state_dict()

        game_runner = partial(run_self_play_game, mcts_rounds, model_state_dict, learning_rate)

        with multiprocessing.Pool(num_workers) as pool:
            results = list(tqdm(pool.imap_unordered(game_runner, range(num_self_play_games))))

        collectors = [collector for pair in results for collector in pair]

        experience = combine_experience(collectors)
        print("Training model...")
        model.train()
        agent_for_training = ZeroAgent(model, encoder)
        agent_for_training.train(experience, batch_size, num_training_epochs)

        if (generation + 1) % model_save_freq == 0:
            print(f"Saving model after generation {generation + 1}")
            torch.save(model.state_dict(), f'model_gen_{generation+1}.pth')
            print("Model saved.")

    torch.save(model.state_dict(), 'model_final.pth')
    print("Training complete. Final model saved.")

if __name__ == "__main__":
    main(num_generations=50, num_self_play_games=200, num_training_epochs=10,
         mcts_rounds=100, batch_size=256, learning_rate=0.001,
         num_workers=5)
