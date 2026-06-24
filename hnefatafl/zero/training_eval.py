"""
Clean, noise-free strength evaluation for the AlphaZero training loop.

Unlike self-play (which uses temperature + Dirichlet noise for exploration) and unlike
`arena_evaluation.py` (which plays via `simulate_game_simple`, and so injects opening Dirichlet
noise even at temp=0), this module plays *greedy* games with NO Dirichlet noise. A short opening
window uses temperature sampling purely to diversify games so repeated matchups aren't identical;
the rest of every game is greedy (argmax visits). That makes the resulting scores a real measure
of learned strength, not of exploratory play.

Score convention: win = 1.0, draw = 0.5, loss = 0.0 (so 0.5 == parity, 1.0 == always wins).
"""
from dataclasses import dataclass

import numpy as np
import torch

from hnefatafl.core.gamestate_v2 import GameState
from hnefatafl.core.gameTypes import Player
from hnefatafl.agents.agent import RandomAgent
from hnefatafl.zero.zeroagent_fast import ZeroAgent


@dataclass
class EvalResult:
    opponent: str
    wins_as_black: int = 0
    losses_as_black: int = 0
    draws_as_black: int = 0
    wins_as_white: int = 0
    losses_as_white: int = 0
    draws_as_white: int = 0

    @property
    def games(self) -> int:
        return (self.wins_as_black + self.losses_as_black + self.draws_as_black
                + self.wins_as_white + self.losses_as_white + self.draws_as_white)

    @property
    def score(self) -> float:
        wins = self.wins_as_black + self.wins_as_white
        draws = self.draws_as_black + self.draws_as_white
        return (wins + 0.5 * draws) / max(1, self.games)

    @property
    def score_as_black(self) -> float:
        g = self.wins_as_black + self.losses_as_black + self.draws_as_black
        return (self.wins_as_black + 0.5 * self.draws_as_black) / max(1, g)

    @property
    def score_as_white(self) -> float:
        g = self.wins_as_white + self.losses_as_white + self.draws_as_white
        return (self.wins_as_white + 0.5 * self.draws_as_white) / max(1, g)

    def summary(self) -> str:
        return (f"vs {self.opponent}: score={self.score:.3f} "
                f"(as black {self.score_as_black:.3f}, as white {self.score_as_white:.3f}) "
                f"over {self.games} games")


def _make_zero_agent(model, encoder, mcts_rounds):
    # No Dirichlet noise: this is an evaluation agent, not a self-play (exploring) agent.
    return ZeroAgent(model, encoder, rounds_per_move=mcts_rounds, c=np.sqrt(2),
                     dirichlet_alpha=0.0, dirichlet_epsilon=0.0)


def play_eval_game(black_agent, white_agent, board_size, max_moves,
                   opening_moves=15, opening_temp=1.0):
    """Play one game and return the winner (Player or None). Greedy after the opening window."""
    game = GameState.new_game(board_size=board_size, max_moves=max_moves)
    agents = {Player.black: black_agent, Player.white: white_agent}
    move_num = 0
    while not game.is_over():
        temp = opening_temp if move_num < opening_moves else 0.0
        move = agents[game.next_player].select_move(game, temperature=temp, add_noise=False)
        if move is None:
            break
        game = game.apply_move(move)
        move_num += 1
    return game.winner


def _record(result: EvalResult, model_is_black: bool, winner):
    if model_is_black:
        if winner == Player.black:
            result.wins_as_black += 1
        elif winner == Player.white:
            result.losses_as_black += 1
        else:
            result.draws_as_black += 1
    else:
        if winner == Player.white:
            result.wins_as_white += 1
        elif winner == Player.black:
            result.losses_as_white += 1
        else:
            result.draws_as_white += 1


def evaluate(model, encoder, make_opponent, label, n_games_per_color=6,
             model_mcts_rounds=100, max_moves=200, board_size=11):
    """
    Play `model` greedily against an opponent as BOTH colors and return an EvalResult.

    make_opponent: zero-arg callable returning a fresh Agent each game (so per-agent state
                   caches don't bleed across games).
    """
    model.eval()
    torch.set_grad_enabled(False)
    result = EvalResult(opponent=label)

    # model as black
    for _ in range(n_games_per_color):
        model_agent = _make_zero_agent(model, encoder, model_mcts_rounds)
        winner = play_eval_game(model_agent, make_opponent(), board_size, max_moves)
        _record(result, model_is_black=True, winner=winner)

    # model as white
    for _ in range(n_games_per_color):
        model_agent = _make_zero_agent(model, encoder, model_mcts_rounds)
        winner = play_eval_game(make_opponent(), model_agent, board_size, max_moves)
        _record(result, model_is_black=False, winner=winner)

    return result


def evaluate_vs_random(model, encoder, n_games_per_color=6, model_mcts_rounds=100,
                       max_moves=200, board_size=11):
    """Sanity benchmark: a model with real knowledge should score ~1.0 here."""
    return evaluate(model, encoder, make_opponent=lambda: RandomAgent(), label="random",
                    n_games_per_color=n_games_per_color, model_mcts_rounds=model_mcts_rounds,
                    max_moves=max_moves, board_size=board_size)


def evaluate_vs_model(model, opponent_model, encoder, n_games_per_color=6,
                      model_mcts_rounds=100, opponent_mcts_rounds=None,
                      max_moves=200, board_size=11, label="previous"):
    """Relative-strength benchmark vs a frozen earlier model. >0.5 means `model` is stronger."""
    opp_rounds = opponent_mcts_rounds or model_mcts_rounds
    opponent_model.eval()
    return evaluate(model, encoder,
                    make_opponent=lambda: _make_zero_agent(opponent_model, encoder, opp_rounds),
                    label=label, n_games_per_color=n_games_per_color,
                    model_mcts_rounds=model_mcts_rounds, max_moves=max_moves, board_size=board_size)


if __name__ == "__main__":
    # Standalone: evaluate the most recent checkpoint of a run vs RandomAgent, right now.
    #   python -m hnefatafl.zero.training_eval [mcts_rounds] [games_per_color] [max_moves]
    # Re-run with a high mcts_rounds (e.g. 800) to test whether the policy is genuinely passive
    # or whether shallow search was just hiding decent play.
    import sys
    from pathlib import Path
    from hnefatafl.encoders.advanced_encoder import SevenPlaneEncoder
    from hnefatafl.zero.lightning_network import DualNetwork

    mcts_rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    games_per_color = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    max_moves = int(sys.argv[3]) if len(sys.argv) > 3 else 200

    project_root = Path(__file__).resolve().parents[1]
    ckpt_dir = project_root / "model" / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not ckpts:
        raise SystemExit(f"No checkpoints found in {ckpt_dir}")

    latest = ckpts[0]
    print(f"Evaluating {latest.name} @ {mcts_rounds} sims, {games_per_color} games/color, max_moves={max_moves}")
    encoder = SevenPlaneEncoder(board_size=11, max_moves=max_moves)
    model = DualNetwork.load_from_checkpoint(checkpoint_path=str(latest), map_location="cpu", encoder=encoder)
    model.cpu().eval()
    torch.set_grad_enabled(False)

    res = evaluate_vs_random(model, encoder, n_games_per_color=games_per_color,
                             model_mcts_rounds=mcts_rounds, max_moves=max_moves)
    print(res.summary())
    print(f"  detail: black W/L/D={res.wins_as_black}/{res.losses_as_black}/{res.draws_as_black}  "
          f"white W/L/D={res.wins_as_white}/{res.losses_as_white}/{res.draws_as_white}")
