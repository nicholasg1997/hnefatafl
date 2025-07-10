import torch
import numpy as np
from hnefatafl.core.gameState import GameState
from hnefatafl.core.gameTypes import Player

def clip_probs(probs, max_prob=0.999):
    clipped_probs =  torch.clamp(probs, min=0, max=max_prob)
    return clipped_probs / clipped_probs.sum(dim=-1, keepdim=True)

def simulate_game(black_player, white_player):
    game = GameState.new_game()
    agents = {
        Player.black: black_player,
        Player.white: white_player
    }
    while not game.is_over():
        next_move = agents[game.next_player].select_move(game)
        game = game.apply_move(next_move)
    winner = game.winner
    return winner