import multiprocessing
from functools import partial

from hnefatafl.encoders.advanced_encoder import SevenPlaneEncoder
from hnefatafl.zero.zeroagent_fast import ZeroAgent
from hnefatafl.zero.network import DualNetwork
from hnefatafl.zero.experiencecollector import ZeroExperienceCollector, combine_experience
from hnefatafl.agents.agent import RandomAgent
from hnefatafl.core.gameTypes import Player
from hnefatafl.utils.nnTrainingUtils import simulate_game
from tqdm import tqdm

import torch
BOARD_SIZE = 11

def run_self_play_game(mcts_rounds, model_state_dict, learning_rate, _):
    board_size = BOARD_SIZE
    encoder = SevenPlaneEncoder(board_size)
    model = DualNetwork(encoder, learning_rate=learning_rate)
    model.load_state_dict(model_state_dict)
    model.eval()

    black_agent = ZeroAgent(model, encoder, rounds_per_move=mcts_rounds, mcts_batch_size=25)
    white_agent = ZeroAgent(model, encoder, rounds_per_move=mcts_rounds, mcts_batch_size=25)

    c1 = ZeroExperienceCollector()
    c2 = ZeroExperienceCollector()
    black_agent.set_collector(c1)
    white_agent.set_collector(c2)

    c1.begin_episode()
    c2.begin_episode()

    winner = simulate_game(black_agent, white_agent, board_size=board_size)

    if winner == Player.black:
        print("Black wins!")
        c1.complete_episode(1.0)
        c2.complete_episode(-1.0)
    elif winner == Player.white:
        print("White wins!")
        c1.complete_episode(-1.0)
        c2.complete_episode(1.0)
    else:
        print("Draw!")
        c1.complete_episode(0.0)
        c2.complete_episode(0.0)
    return c1, c2

# create function to analyze agent performance. test against another agent (either random or previously trained) 25 times as white and 25 times as black. also print one game of current agent vs itself
def analyze_agent_performance(agent, opponent_agent, num_games=50):
    board_size = BOARD_SIZE
    opp_agent = opponent_agent
    a1 = agent
    a2 = opp_agent
    win_counts = {Player.black: 0, Player.white: 0, None: 0}
    for i in range(num_games):
        print(f"Game {i + 1}/{num_games}")
        verbose = False
        if i == num_games:
            verbose = True
        winner = simulate_game(a1, a2, board_size=board_size, verbose=verbose)
        if winner == Player.black:
            win_counts[Player.black] += 1
        elif winner == Player.white:
            win_counts[Player.white] += 1
        else:
            win_counts[None] += 1
    print(f"Results after {num_games} games:")
    print(f"Black wins: {win_counts[Player.black]}")
    print(f"White wins: {win_counts[Player.white]}")
    print(f"Draws: {win_counts[None]}")




def main(learning_rate=0.001, batch_size=16, num_generations = 10,
         num_self_play_games=2, num_training_epochs=1, mcts_rounds=25, model_save_freq=5,
         num_workers=None):
    board_size = BOARD_SIZE

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
            torch.save(model.state_dict(), f'models/model_gen_{generation+1}.pth')
            print("Model saved.")
            print("Analyzing agent performance...")
            analyze_agent_performance(agent_for_training, agent_for_training, num_games=5)


    torch.save(model.state_dict(), 'models/model_final.pth')
    print("Training complete. Final model saved.")

if __name__ == "__main__":
    main(num_generations=10, num_self_play_games=200, num_training_epochs=10,
         mcts_rounds=50, batch_size=256, learning_rate=1e-4,
         num_workers=1, model_save_freq=1)

