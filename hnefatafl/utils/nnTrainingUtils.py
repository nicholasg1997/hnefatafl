import torch
import numpy as np
import time

from hnefatafl.core.gamestate_v2 import GameState
from hnefatafl.core.gameTypes import Player

def clip_probs(probs, max_prob=0.999):
    clipped_probs =  torch.clamp(probs, min=0, max=max_prob)
    return clipped_probs / clipped_probs.sum(dim=-1, keepdim=True)

def simulate_game(black_player, white_player, board_size=11, max_moves=500, resign_threshold=0.95, temp= 1.0, verbose=False):
    game = GameState.new_game(board_size=board_size)
    agents = {
        Player.black: black_player,
        Player.white: white_player
    }
    move_count = 0
    while not game.is_over():
        next_move = agents[game.next_player].select_move(game, temp=temp)

        if hasattr(agents[game.next_player], 'last_value_estimate'):
            if abs(agents[game.next_player].last_value_estimate) > resign_threshold:
                winner = game.next_player.other if agents[game.next_player].last_value_estimate < -resign_threshold else None
                if verbose:
                    print(f"{game.next_player} resigns due to high value estimate: {agents[game.next_player].last_value_estimate}")
                return winner

        game = game.apply_move(next_move)
        print(game.board)
        print(f"Move {move_count + 1}: {game.last_move}")
        if verbose:
            print(f"Move {move_count + 1}: {game.last_move}")
            print(game.board)
        move_count += 1
        if move_count > max_moves:
            print("Maximum move limit reached, ending game.")
            print(game.board)
            return None
    winner = game.winner
    return winner

def simulate_game_simple(black_player, white_player, board_size=11, max_moves=250, temp = 1.0, verbose=False):
    game = GameState.new_game(board_size=board_size)
    agents = {
        Player.black: black_player,
        Player.white: white_player
    }
    move_count = 0
    while not game.is_over():
        time_start = time.time()
        is_exploring = move_count < 30
        temperature = temp if is_exploring else 0.1
        add_noise = is_exploring

        next_move = agents[game.next_player].select_move(game, temperature=temperature, add_noise=add_noise)

        if next_move is None:
            print("no legal moves available, ending game.")
            return game.winner

        game = game.apply_move(next_move)
        if verbose:
            print(f"Move {move_count + 1}: {game.last_move} by {game.next_player.other}")
            print(game.board)
        move_count += 1
        end_time = time.time()
        if verbose:
            print(f"Move {move_count}: {game.last_move} took {end_time - time_start:.2f} seconds")
        if move_count > max_moves:
            print("Maximum move limit reached, ending game.")
            return game
    winner = game.winner
    print(f"Game ended in {move_count} moves. Winner: {winner}, duplication detected:{game.repetition_hit}")
    return game

if __name__ == "__main__":
    from hnefatafl.encoders.advanced_encoder import SevenPlaneEncoder
    from hnefatafl.zero.zeroagent_v2 import ZeroAgent
    #from hnefatafl.zero.zeroagent_fast import ZeroAgent
    from hnefatafl.zero.network import DualNetwork
    from hnefatafl.agents.agent import RandomAgent

    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    ckpt_path = project_root / "zero" / "lightning_logs" / "version_2" / "checkpoints" / "epoch=4-step=5860.ckpt"

    encoder = SevenPlaneEncoder(11)
    #model = DualNetwork.load_from_checkpoint(ckpt_path, encoder=encoder)
    model = DualNetwork(encoder)
    model = model.to("cpu")
    model.eval()
    alpha_agent = ZeroAgent(model, encoder, rounds_per_move=200, c=0, dirichlet_alpha=0.0, dirichlet_epsilon=0.0)
    winner = simulate_game_simple(alpha_agent, RandomAgent(), verbose=True, max_moves=200, temp=0.1)
    if winner is None:
        print("Game ended in a draw.")
    elif winner == Player.black:
        print("Black wins!")
    else:
        print("White wins!")