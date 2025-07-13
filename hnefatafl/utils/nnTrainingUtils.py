import torch
import numpy as np
from hnefatafl.core.gameState import GameState
from hnefatafl.core.gameTypes import Player

def clip_probs(probs, max_prob=0.999):
    clipped_probs =  torch.clamp(probs, min=0, max=max_prob)
    return clipped_probs / clipped_probs.sum(dim=-1, keepdim=True)

def simulate_game(black_player, white_player, max_moves=1000, verbose=False):
    game = GameState.new_game()
    agents = {
        Player.black: black_player,
        Player.white: white_player
    }
    move_count = 0
    while not game.is_over():
        next_move = agents[game.next_player].select_move(game)
        game = game.apply_move(next_move)
        if verbose:
            print(f"Move {move_count + 1}: {game.last_move}")
            print(game.board)
        move_count += 1
        if move_count > max_moves:
            return None
    winner = game.winner
    return winner

if __name__ == "__main__":
    from hnefatafl.encoders.advanced_encoder import SevenPlaneEncoder
    from hnefatafl.zero.zeroagent import ZeroAgent
    from hnefatafl.zero.network import DualNetwork

    encoder = SevenPlaneEncoder(11)
    model = DualNetwork(encoder)
    model.load_state_dict(torch.load('/Users/nickgault/PycharmProjects/hnefatafl/hnefatafl/zero/model.pth'))
    model.eval()
    black_agent = ZeroAgent(model, encoder, rounds_per_move=25)
    white_agent = ZeroAgent(model, encoder, rounds_per_move=25)
    winner = simulate_game(black_agent, white_agent, verbose=True)
    if winner is None:
        print("Game ended in a draw.")
    elif winner == Player.black:
        print("Black wins!")
    else:
        print("White wins!")