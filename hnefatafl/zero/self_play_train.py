import multiprocessing
import os
from functools import partial
from itertools import chain
from tqdm import tqdm
import torch

from hnefatafl.encoders.advanced_encoder import SevenPlaneEncoder
from hnefatafl.zero.zeroagent_v2 import ZeroAgent
from hnefatafl.zero.network import DualNetwork
from hnefatafl.zero.experiencecollector import ZeroExperienceCollector, combine_experience
from hnefatafl.core.gameTypes import Player
from hnefatafl.utils.nnTrainingUtils import simulate_game_simple as simulate_game

def run_self_play_game(model_state_dict, encoder, mcts_rounds, _):
    """
    Run a self-play game using the provided model state and encoder.
    Args:
        model_state_dict (dict): The state dictionary of the model to be used.
        encoder (SevenPlaneEncoder): The encoder for the game state.
        mcts_rounds (int): Number of MCTS rounds to perform per move.
    Returns:
        tuple: Two ZeroExperienceCollector instances for black and white agents.
    """
    model = DualNetwork(encoder)
    model.load_state_dict(model_state_dict)

    black_agent = ZeroAgent(model, encoder, rounds_per_move=mcts_rounds, c=4)
    white_agent = ZeroAgent(model, encoder, rounds_per_move=mcts_rounds, c=4)

    c1 = ZeroExperienceCollector()
    c2 = ZeroExperienceCollector()
    black_agent.set_collector(c1)
    white_agent.set_collector(c2)

    c1.begin_episode()
    c2.begin_episode()

    winner = simulate_game(black_agent, white_agent, max_moves=200, verbose=True)

    if winner == Player.black:
        c1.complete_episode(1.0)
        c2.complete_episode(-1.0)
    elif winner == Player.white:
        c1.complete_episode(-1.0)
        c2.complete_episode(1.0)
    else:  # Draw
        c1.complete_episode(-0.5)
        c2.complete_episode(-0.5)

    return c1, c2


def main(learning_rate=0.01, batch_size=16, num_generations=10,
         num_self_play_games=2, num_training_epochs=1, mcts_rounds=25, model_save_freq=10):
    board_size = 11
    num_generations = num_generations
    learning_rate = learning_rate
    batch_size = batch_size
    num_self_play_games = num_self_play_games
    num_training_epochs = num_training_epochs
    mcts_rounds = mcts_rounds
    model_save_freq = model_save_freq

    free_cores = 9
    num_workers = max(1, os.cpu_count() - free_cores) # leave n cores free to keep Mac cool
    print(f"Using {num_workers} worker processes for self-play.")

    encoder = SevenPlaneEncoder(board_size)
    model = DualNetwork(encoder, learning_rate=learning_rate)

    for generation in range(num_generations):
        print(f"Starting generation {generation + 1}/{num_generations}")

        model.cpu()
        model.eval()
        model_state_dict = model.state_dict()

        with multiprocessing.Pool(processes=num_workers) as pool:
            with tqdm(total=num_self_play_games, desc="Simulating games") as pbar:
                results = []
                for result in pool.imap_unordered(partial(run_self_play_game, model_state_dict, encoder, mcts_rounds),
                                                  range(num_self_play_games)):
                    results.append(result)
                    pbar.update(1)

        collectors = list(chain.from_iterable(results))

        experience = combine_experience(collectors)
        print("Training model...")
        model.train()
        training_agent = ZeroAgent(model, encoder)
        training_agent.train(experience, batch_size, num_training_epochs)
        print("Training complete.")

        if (generation + 1) % model_save_freq == 0:
            print(f"Saving model after generation {generation + 1}")
            torch.save(model.state_dict(), f'models/model_gen_{generation + 1}.pth')
            print("Model saved.")
            # TODO: implement model evaluation against a baseline (random/previous agent) and save best agent.
            #  number of games completed, average game length, win rate by color, etc.
            simulate_game(training_agent, training_agent, max_moves=150, verbose=True)

    torch.save(model.state_dict(), 'models/model_final.pth')
    print("Training complete. Final model saved.")


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main(num_generations=20, num_self_play_games=200, num_training_epochs=12, mcts_rounds=300, batch_size=128,
         learning_rate=1e-4)