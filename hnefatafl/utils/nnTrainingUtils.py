import torch
import time
from dataclasses import dataclass
import numpy as np

from hnefatafl.core.gamestate_v2 import GameState
from hnefatafl.core.gameTypes import Player

def clip_probs(probs, max_prob=0.999):
    clipped_probs =  torch.clamp(probs, min=0, max=max_prob)
    return clipped_probs / clipped_probs.sum(dim=-1, keepdim=True)

@dataclass(frozen=True)
class ProgressiveMCTSConfigs:
    depths: list[int]
    initial_probs: list[float]
    final_probs: list[float]
    total_gens: int

    def __post_init__(self):
        if not (len(self.depths) == len(self.initial_probs) == len(self.final_probs)):
            raise ValueError("depths, initial_probs, and final_probs must have the same length.")

        for name, probs in ("initial_probs", self.initial_probs), ("final_probs", self.final_probs):
            s = sum(probs)
            if not np.isclose(s, 1.0, atol=1e-8):
                raise ValueError(f"{name} must sum to 1.0, but got {s:.6f}.")

    def sample(self, gen, schedule="linear"):
        gen = min(max(gen, 0), self.total_gens) # Ensure generation is greater than 0 but less than total_gens
        t = gen / self.total_gens

        if schedule == "exp":
            alpha = 1 - np.exp(-5*t)
        elif schedule == "log":
            alpha = np.log1p(t * (np.e - 1))
        elif schedule == "quadratic":
            alpha = t ** 2
        elif schedule == "linear":
            alpha = t
        else:
            print(f"unknown scheduler {schedule}, using linear schedule.")
            alpha = t

        adjusted_probs = (1-alpha) * np.array(self.initial_probs) + alpha * np.array(self.final_probs)
        adjusted_probs = adjusted_probs / adjusted_probs.sum()
        return int(np.random.choice(self.depths, p=adjusted_probs))

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
    game = GameState.new_game(board_size=board_size, max_moves=max_moves)
    agents = {
        Player.black: black_player,
        Player.white: white_player
    }
    move_count = 0
    while not game.is_over():
        time_start = time.time()
        is_exploring = move_count < 50
        temperature = temp if is_exploring else min(0.5, temp)
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
        if move_count + 1 > max_moves:
            print("Maximum move limit reached, ending game now.")
            return game
    print(f"Game ended in {move_count} moves. Winner: {game.winner}, duplication detected:{game.repetition_hit}")
    print(game.board)
    return game

if __name__ == "__main__":
    from hnefatafl.encoders.advanced_encoder import SevenPlaneEncoder
    #from hnefatafl.zero.zeroagent_v2 import ZeroAgent
    from hnefatafl.zero.zeroagent_fast import ZeroAgent
    from hnefatafl.zero.network import DualNetwork
    from hnefatafl.agents.agent import RandomAgent

    import cProfile
    import pstats

    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    ckpt_path = project_root / "zero" / "lightning_logs" / "version_16" / "checkpoints" / "epoch=2-step=684.ckpt"
    ckpt_path = "/Users/nickgault/PycharmProjects/hnefatafl/hnefatafl/models/checkpoints/model-epoch=02-total_loss=3.28.ckpt"
    profiler = cProfile.Profile()

    encoder = SevenPlaneEncoder(11)
    model = DualNetwork.load_from_checkpoint(ckpt_path, encoder=encoder)
    #model = DualNetwork(encoder)
    model = model.to("cpu")
    model.eval()

    profiler.enable()
    alpha_agent = ZeroAgent(model, encoder, rounds_per_move=400, c=1.8, dirichlet_alpha=0.0, dirichlet_epsilon=0.0)
    end_game = simulate_game_simple(alpha_agent, alpha_agent, verbose=True, max_moves=200, temp=0.0)
    winner = end_game.winner
    if winner is None:
        print("Game ended in a draw.")
    elif winner == Player.black:
        print("Black wins!")
    else:
        print("White wins!")

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.print_stats(20)