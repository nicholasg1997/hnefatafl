import gc
import multiprocessing
import os
from functools import partial
from itertools import chain
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
from tqdm import tqdm

from hnefatafl.core.gameTypes import Player
from hnefatafl.encoders.advanced_encoder import SevenPlaneEncoder
from hnefatafl.utils.nnTrainingUtils import simulate_game_simple as simulate_game
from hnefatafl.utils.nnTrainingUtils import ProgressiveMCTSConfigs
from hnefatafl.utils.statTracker import StatTracker
from hnefatafl.zero.experienceCollector_v2 import ZeroExperienceCollector, PersistentExperienceBuffer, \
    ReplayBufferCheckpoint
from hnefatafl.zero.lightning_network import DualNetwork
from hnefatafl.zero.zeroagent_fast import ZeroAgent

def _run_self_play_game_worker(model_state_dict, encoder, sampler, current_gen, max_moves, _):
    step_penalty = 0.001

    temp_model = DualNetwork(encoder)
    temp_model.load_state_dict(model_state_dict)
    temp_model.to('cpu')
    temp_model.eval()
    torch.set_grad_enabled(False)

    mcts_rounds = sampler.sample(current_gen)
    print(f"MCTS rounds: {mcts_rounds}")

    black_agent = ZeroAgent(temp_model, encoder, rounds_per_move=mcts_rounds, c=np.sqrt(2))
    white_agent = ZeroAgent(temp_model, encoder, rounds_per_move=mcts_rounds, c=np.sqrt(2))

    c1 = ZeroExperienceCollector()
    c2 = ZeroExperienceCollector()
    black_agent.set_collector(c1)
    white_agent.set_collector(c2)

    c1.begin_episode()
    c2.begin_episode()

    game = simulate_game(black_agent, white_agent, max_moves=max_moves, verbose=False, board_size=11)
    winner = game.winner
    time_penalty = step_penalty * game.move_count

    if winner == Player.black:
        c1.complete_episode(1.0 - time_penalty, is_result=True)
        c2.complete_episode(-1.0 + time_penalty, is_result=True)
    elif winner == Player.white:
        c1.complete_episode(-1.0 + time_penalty, is_result=True)
        c2.complete_episode(1.0 - time_penalty, is_result=True)
    else:
        c1.complete_episode(0.0, is_result=False)
        c2.complete_episode(0.0, is_result=False)

    game_stats = {
        "winner": winner,
        "move_count": game.move_count,
        "repetition_hit": game.repetition_hit,
        "repeating_player": getattr(game, "repeating_player", None),
        "move_limit_hit": game.move_limit_hit,
    }
    return c1, c2, game_stats

#TODO: convert all the parameters to a config object
def main(num_generations=20, num_self_play_games=200, num_training_epochs=3,
         batch_size=128, learning_rate=0.0001, max_moves=500):
    project_root = Path(__file__).resolve().parents[1]
    checkpoints_dir = project_root / "model" / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    encoder = SevenPlaneEncoder(board_size=11)
    current_model = DualNetwork(encoder=encoder, learning_rate=learning_rate, batch_size=batch_size)
    persistent_buffer = PersistentExperienceBuffer(max_games=150_000)
    stat_tracker = StatTracker()

    start_generation = 0
    wandb_run_id = None
    latest_ckpt_path = checkpoints_dir / "last-v4.ckpt"

    if latest_ckpt_path.exists():
        print(f"Found checkpoint. Resuming training from: {latest_ckpt_path}")
        full_ckpt = torch.load(latest_ckpt_path, weights_only=False)
        current_model.load_state_dict(full_ckpt['state_dict'])

        replay_callback = ReplayBufferCheckpoint(persistent_buffer)
        replay_callback.on_load_checkpoint(None, None, full_ckpt)

        start_generation = full_ckpt.get('current_generation', 0)
        wandb_run_id = full_ckpt.get('wandb_run_id')
        print(f"Resuming from generation {start_generation} with run ID {wandb_run_id}")
    else:
        print("No checkpoint found. Starting new training run.")
        latest_ckpt_path = None

    mcts_config = ProgressiveMCTSConfigs(
        depths=[200, 400, 800, 1200],
        initial_probs=[0.5, 0.4, 0.1, 0.0],
        final_probs=[0.1, 0.3, 0.55, 0.05],
        total_gens=num_generations
    )

    for generation in range(start_generation, num_generations):
        gc.collect()
        print(f"\n===== STARTING GENERATION {generation + 1}/{num_generations} =====")
        print(f"Current buffer size: {len(persistent_buffer)}")

        print("--- Generating self-play games... ---")
        current_generation = generation + 1
        current_model.eval()
        torch.set_grad_enabled(False)
        model_state_dict = current_model.state_dict()
        free_cores = 2
        num_workers = max(1, os.cpu_count() - free_cores)

        with multiprocessing.get_context("spawn").Pool(processes=num_workers) as pool:
            with tqdm(total=num_self_play_games, desc=f"Gen {current_generation} Games") as pbar:
                worker_func = partial(_run_self_play_game_worker,
                                      model_state_dict,
                                      encoder,
                                      mcts_config,
                                      generation,
                                      max_moves)
                all_results = []
                for result in pool.imap_unordered(worker_func, range(num_self_play_games)):
                    all_results.append(result)
                    pbar.update(1)

        collectors = [res[0] for res in all_results] + [res[1] for res in all_results]
        game_stats_list = [res[2] for res in all_results]

        persistent_buffer.add_experience(collectors)
        for stats in game_stats_list:
            stat_tracker.log_game(stats)

        print(f"Buffer size: {len(persistent_buffer)}")

        print("--- Training on new data... ---")
        current_model.train()
        torch.set_grad_enabled(True)

        training_experience = persistent_buffer.get_training_buffer()
        if len(training_experience.states) < batch_size:
            print("Not enough data in buffer to train. Skipping training for this generation.")
            continue

        train_loader = training_experience.get_dataloader(batch_size=batch_size)

        current_model.current_generation = current_generation

        wandb_logger = WandbLogger(project="hnefatafl-zero", id=wandb_run_id, resume="allow")
        wandb_run_id = wandb_logger.experiment.id

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=checkpoints_dir,
            filename=f"model-gen_{current_generation}" + "-{epoch:02d}",
            save_last=True,
            save_top_k=0,
        )

        trainer = pl.Trainer(
            max_epochs=num_training_epochs,
            logger=wandb_logger,
            callbacks=[ReplayBufferCheckpoint(persistent_buffer), checkpoint_callback],
            enable_progress_bar=True,
            log_every_n_steps=10
        )

        trainer.fit(current_model, train_dataloaders=train_loader)

        stat_tracker.summarize_generation(current_generation)

    print("Training complete.")
    stat_tracker.close()


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main(num_generations=20, num_self_play_games=200, num_training_epochs=5,
         batch_size=128, learning_rate=0.0001, max_moves=400)
