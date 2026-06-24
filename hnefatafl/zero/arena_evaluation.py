import os
import csv
from tqdm import tqdm
import shutil
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from functools import partial
from multiprocessing import get_context
from typing import Tuple, List, Optional

import torch
import numpy as np

from hnefatafl.zero.zeroagent_fast import ZeroAgent
from hnefatafl.utils.nnTrainingUtils import simulate_game_simple as simulate_game
from hnefatafl.core.gameTypes import Player
from hnefatafl.encoders.advanced_encoder import SevenPlaneEncoder
from hnefatafl.zero.lightning_network import DualNetwork


@dataclass
class ArenaConfig:
    board_size: int = 11
    n_games_total: int = 40
    mcts_rounds: int = 300
    max_moves: int = 500
    device: str = "cpu"
    num_workers: int = max(1, (os.cpu_count() or 4) - 2)
    top_n: int = 5
    win_threshold: float = 0.55
    min_color_rate: float = 0.50
    arena_dir: str = "arena"
    top_models_dir: str = "arena/top_models"
    leaderboard_csv: str = "arena/leaderboard.csv"
    champion_name: str = "champion.pt"


@dataclass
class ArenaResult:
    timestamp: float
    challenger_path: str
    vs_champion_path: Optional[str]
    wins_as_white: int
    wins_as_black: int
    draws_as_white: int
    draws_as_black: int
    games_as_white: int
    games_as_black: int

    @property
    def win_rate_white(self) -> float:
        return (self.wins_as_white + 0.5 * self.draws_as_white) / max(1, self.games_as_white)

    @property
    def win_rate_black(self) -> float:
        return (self.wins_as_black + 0.5 * self.draws_as_black) / max(1, self.games_as_black)

    @property
    def win_rate_overall(self) -> float:
        total_games = self.games_as_white + self.games_as_black
        total_points = (
            self.wins_as_white + self.wins_as_black
            + 0.5 * (self.draws_as_white + self.draws_as_black)
        )
        return total_points / max(1, total_games)


def _load_model_from_ckpt(ckpt_path: Path, encoder: SevenPlaneEncoder, device: str) -> DualNetwork:
    """Load a full PyTorch Lightning checkpoint."""
    model = DualNetwork.load_from_checkpoint(
        checkpoint_path=str(ckpt_path),
        map_location=device,
        encoder=encoder,
        weights_only=False
    )
    model.to(device)
    model.eval()
    torch.set_grad_enabled(False)
    return model


def _play_one_game(
    a_path: Path,
    b_path: Path,
    a_is_black: bool,
    encoder: SevenPlaneEncoder,
    mcts_rounds: int,
    max_moves: int,
    device: str,
) -> int:
    """
    Returns:
        +1 if A wins, 0 if draw, -1 if A loses.
    """
    model_a = _load_model_from_ckpt(a_path, encoder, device)
    model_b = _load_model_from_ckpt(b_path, encoder, device)

    agent_a = ZeroAgent(model_a, encoder, rounds_per_move=mcts_rounds, c=np.sqrt(2))
    agent_b = ZeroAgent(model_b, encoder, rounds_per_move=mcts_rounds, c=np.sqrt(2))

    if a_is_black:
        black_agent, white_agent = agent_a, agent_b
    else:
        black_agent, white_agent = agent_b, agent_a

    game = simulate_game(black_agent, white_agent, max_moves=max_moves, verbose=False, temp=0)
    winner = game.winner

    if winner == Player.black:
        a_won = +1 if a_is_black else -1
    elif winner == Player.white:
        a_won = +1 if not a_is_black else -1
    else:
        a_won = 0
    return a_won


def _worker_play_chunk(
    jobs: List[Tuple[str, str, bool]],
    encoder: SevenPlaneEncoder,
    mcts_rounds: int,
    max_moves: int,
    device: str,
) -> Tuple[int, int, int, int]:
    """
    A worker processes a chunk of games to amortize model loads.
    Returns:
        wins_as_white, wins_as_black, draws_as_white, draws_as_black
    """
    wins_w = wins_b = draws_w = draws_b = 0
    for a_path, b_path, a_is_black in jobs:
        outcome = _play_one_game(
            Path(a_path), Path(b_path), a_is_black,
            encoder, mcts_rounds, max_moves, device
        )
        if a_is_black:
            if outcome > 0: wins_b += 1
            elif outcome == 0: draws_b += 1
        else:
            if outcome > 0: wins_w += 1
            elif outcome == 0: draws_w += 1
    return wins_w, wins_b, draws_w, draws_b


class Arena:
    def __init__(self, cfg: ArenaConfig):
        self.cfg = cfg
        self.arena_dir = Path(self.cfg.arena_dir)
        self.top_models_dir = Path(self.cfg.top_models_dir)
        self.leaderboard_csv = Path(self.cfg.leaderboard_csv)
        self.arena_dir.mkdir(parents=True, exist_ok=True)
        self.top_models_dir.mkdir(parents=True, exist_ok=True)
        self.encoder = SevenPlaneEncoder(board_size=self.cfg.board_size)

    @property
    def champion_path(self) -> Path:
        return self.top_models_dir / self.cfg.champion_name

    def _append_leaderboard(self, result: ArenaResult) -> None:
        csv_path = Path(self.cfg.leaderboard_csv)
        is_new = not csv_path.exists()
        with csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(asdict(result).keys()) + [
                    "win_rate_white", "win_rate_black", "win_rate_overall"
                ],
            )
            if is_new:
                writer.writeheader()
            row = asdict(result)
            row.update({
                "win_rate_white": f"{result.win_rate_white:.4f}",
                "win_rate_black": f"{result.win_rate_black:.4f}",
                "win_rate_overall": f"{result.win_rate_overall:.4f}",
            })
            writer.writerow(row)

    def _promote_champion(self, challenger_ckpt: Path):
        shutil.copy2(challenger_ckpt, self.champion_path)
        stamped = self.top_models_dir / f"{challenger_ckpt.stem}-{int(time.time())}.ckpt"
        shutil.copy2(challenger_ckpt, stamped)
        self._prune_top_n()

    def _prune_top_n(self) -> None:
        files = sorted(
            [p for p in Path(self.cfg.top_models_dir).glob("*.ckpt") if p.name != self.cfg.champion_name],
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        for extra in files[self.cfg.top_n:]:
            extra.unlink(missing_ok=True)

    def _build_job_list(self, challenger_path: Path, champion_path: Path) -> List[Tuple[str, str, bool]]:
        n_per_color = self.cfg.n_games_total // 2
        jobs = []
        for _ in range(n_per_color): jobs.append((str(challenger_path), str(champion_path), True))
        for _ in range(n_per_color): jobs.append((str(challenger_path), str(champion_path), False))
        return jobs

    def evaluate_challenger(self, challenger_ckpt: Path):
        if not self.champion_path.exists():
            print("[Arena] No champion yet — promoting challenger automatically.")
            self._promote_champion(challenger_ckpt)
            return

        jobs = self._build_job_list(challenger_ckpt, self.champion_path)
        chunks = [jobs[i::self.cfg.num_workers] for i in range(self.cfg.num_workers)]

        ctx = get_context("spawn")
        worker = partial(_worker_play_chunk, encoder=self.encoder, mcts_rounds=self.cfg.mcts_rounds,
                         max_moves=self.cfg.max_moves, device=self.cfg.device)

        totals = {'w_w': 0, 'w_b': 0, 'd_w': 0, 'd_b': 0}
        with ctx.Pool(processes=self.cfg.num_workers) as pool:
            for res in tqdm(pool.imap_unordered(worker, chunks), total=len(chunks), desc="Arena Chunks"):
                totals['w_w'] += res[0]
                totals['w_b'] += res[1]
                totals['d_w'] += res[2]
                totals['d_b'] += res[3]

        result = ArenaResult(
            timestamp=time.time(), challenger_path=str(challenger_ckpt),
            vs_champion_path=str(self.champion_path), wins_as_white=totals["w_w"],
            wins_as_black=totals["w_b"], draws_as_white=totals["d_w"],
            draws_as_black=totals["d_b"], games_as_white=self.cfg.n_games_total // 2,
            games_as_black=self.cfg.n_games_total // 2,
        )

        print(f"[Arena] Results — Challenger vs Champion | "
              f"white: {result.win_rate_white:.3f} | black: {result.win_rate_black:.3f} | "
              f"overall: {result.win_rate_overall:.3f}")

        if (result.win_rate_overall >= self.cfg.win_threshold and
                result.win_rate_white >= self.cfg.min_color_rate and
                result.win_rate_black >= self.cfg.min_color_rate):
            print("NEW CHAMPION! Promoting challenger.")
            self._promote_champion(challenger_ckpt)
        else:
            print("[Arena] Challenger rejected.")

        self._append_leaderboard(result)