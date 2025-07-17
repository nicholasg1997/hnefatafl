import torch
import numpy as np

from hnefatafl.agents.agent import RandomAgent
from hnefatafl.core.gameState import GameState
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
        is_exploring = move_count < 30
        temperature = temp if is_exploring else 0.0
        add_noise = is_exploring

        next_move = agents[game.next_player].select_move(game, temperature=temperature, add_noise=add_noise)

        if next_move is None:
            print("no legal moves available, ending game.")
            return game.winner

        game = game.apply_move(next_move)
        if verbose:
            print(f"Move {move_count + 1}: {game.last_move}")
            print(game.board)
        move_count += 1
        if move_count > max_moves:
            print("Maximum move limit reached, ending game.")
            return None
    winner = game.winner
    print(f"Game ended in {move_count} moves. Winner: {winner}")
    return winner

if __name__ == "__main__":
    from hnefatafl.encoders.advanced_encoder import SevenPlaneEncoder
    from hnefatafl.zero.zeroagent import ZeroAgent
    from hnefatafl.zero.network import DualNetwork

    encoder = SevenPlaneEncoder(11)
    model = RandomAgent()
    #model.load_state_dict(torch.load('/Users/nickgault/PycharmProjects/hnefatafl/hnefatafl/zero/model.pth'))
    #model.eval()
    #black_agent = ZeroAgent(model, encoder, rounds_per_move=25)
    #white_agent = ZeroAgent(model, encoder, rounds_per_move=25)
    winner = simulate_game(model, model, verbose=True)
    if winner is None:
        print("Game ended in a draw.")
    elif winner == Player.black:
        print("Black wins!")
    else:
        print("White wins!")