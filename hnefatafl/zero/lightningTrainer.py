import gc
import multiprocessing
import os
from functools import partial
from pathlib import Path
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
from tqdm import tqdm
import pickle

from hnefatafl.core.gameTypes import Player
from hnefatafl.encoders.advanced_encoder import SevenPlaneEncoder
from hnefatafl.utils.nnTrainingUtils import simulate_game_simple as simulate_game
from hnefatafl.utils.nnTrainingUtils import ProgressiveMCTSConfigs
from hnefatafl.utils.statTracker import StatTracker
from hnefatafl.zero.experienceCollector_v2 import ZeroExperienceCollector, PersistentExperienceBuffer
from hnefatafl.zero.lightning_network import DualNetwork
from hnefatafl.zero.zeroagent_fast import ZeroAgent
from hnefatafl.zero.zeroagent_v2 import ZeroAgent as BatchedZeroAgent
from hnefatafl.zero.training_eval import evaluate_vs_random, evaluate_vs_model

def _finish_game(black_agent, white_agent, max_moves, step_penalty=0.0005, timeout_reward=0.3):
    """Play one self-play game with the given agents and return (c1, c2, game_stats).

    Shared by both the multiprocessing CPU path and the single-process GPU path so their
    reward/stat bookkeeping can never drift apart.

    Reward shaping (to break the all-draw collapse):
      * Decisive win: winner gets +1 - step_penalty*moves, loser the negation -> faster wins
        are worth more, discouraging slow play. The discount is capped so a win stays >= ~0.6.
      * Timeout (move-limit, winner=None): black +timeout_reward / white -timeout_reward. This
        breaks the draw symmetry (white must actually escape or it loses points) while keeping a
        real capture (+1) strictly better than a timeout, so black isn't pushed to pure-stall.
    """
    c1 = ZeroExperienceCollector()
    c2 = ZeroExperienceCollector()
    black_agent.set_collector(c1)
    white_agent.set_collector(c2)
    c1.begin_episode()
    c2.begin_episode()

    game = simulate_game(black_agent, white_agent, max_moves=max_moves, verbose=False, board_size=11)
    winner = game.winner
    length_discount = min(0.4, step_penalty * game.move_count)  # cap keeps decisive wins >= ~0.6

    if winner == Player.black:
        c1.complete_episode(1.0 - length_discount, is_result=True)
        c2.complete_episode(-1.0 + length_discount, is_result=True)
    elif winner == Player.white:
        c1.complete_episode(-1.0 + length_discount, is_result=True)
        c2.complete_episode(1.0 - length_discount, is_result=True)
    else:
        # Timeout / no winner: small asymmetric reward, NOT a decisive result.
        c1.complete_episode(timeout_reward, is_result=False)
        c2.complete_episode(-timeout_reward, is_result=False)

    game_stats = {
        "winner": winner,
        "move_count": game.move_count,
        "repetition_hit": game.repetition_hit,
        "repeating_player": getattr(game, "repeating_player", None),
        "move_limit_hit": game.move_limit_hit,
    }
    return c1, c2, game_stats


def _run_self_play_game_worker(model_state_dict, encoder, sampler, current_gen, max_moves, _):
    # One intra-op thread per worker: we already parallelize across (cpu_count - 2)
    # processes, so letting each process spin up its default thread pool oversubscribes
    # the cores and slows everything down.
    torch.set_num_threads(1)

    temp_model = DualNetwork(encoder)
    temp_model.load_state_dict(model_state_dict)
    temp_model.to('cpu')
    temp_model.eval()
    torch.set_grad_enabled(False)

    mcts_rounds = sampler.sample(current_gen)
    print(f"MCTS rounds: {mcts_rounds}")

    black_agent = ZeroAgent(temp_model, encoder, rounds_per_move=mcts_rounds, c=np.sqrt(2))
    white_agent = ZeroAgent(temp_model, encoder, rounds_per_move=mcts_rounds, c=np.sqrt(2))
    return _finish_game(black_agent, white_agent, max_moves)


def _self_play_in_process(model, encoder, sampler, generation, max_moves, num_games,
                          device, mcts_batch_size):
    """Single-process self-play using the batched MCTS agent (BatchedZeroAgent).

    GPU contexts (MPS/CUDA) can't be shared across spawned worker processes, so when running
    on a GPU device we play games sequentially in this process and rely on batched leaf
    evaluation to keep the device busy. The batched forward pass is where MPS/CUDA actually
    beats CPU (see the per-batch benchmark in the PR notes).
    """
    model.eval().to(device)
    torch.set_grad_enabled(False)
    mcts_rounds = sampler.sample(generation)
    print(f"In-process self-play on '{device}': {mcts_rounds} MCTS rounds, batch={mcts_batch_size}")

    results = []
    for _ in tqdm(range(num_games), desc=f"Gen {generation + 1} Games ({device})"):
        black_agent = BatchedZeroAgent(model, encoder, rounds_per_move=mcts_rounds,
                                       c=np.sqrt(2), batch_size=mcts_batch_size)
        white_agent = BatchedZeroAgent(model, encoder, rounds_per_move=mcts_rounds,
                                       c=np.sqrt(2), batch_size=mcts_batch_size)
        results.append(_finish_game(black_agent, white_agent, max_moves))
    return results

def save_replay_buffer(buffer, path, create_path=True):
    """
    Save the replay buffer to a file.
    :param buffer: The PersistentExperienceBuffer instance to save.
    :param path: The file path where the buffer will be saved.
    :param create_path: If True, create the directory if it does not exist.
    """
    if create_path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, 'wb') as f:
        pickle.dump(buffer.get_states(), f)
    os.replace(tmp_path, path)
    print(f"Replay buffer saved to {path}")

def load_replay_buffer(path):
    """
    Load the replay buffer from a file.
    :param path: The file path from which to load the buffer.
    :return: A PersistentExperienceBuffer instance with the loaded data.
    """
    with open(path, 'rb') as f:
        load_states = pickle.load(f)
        return load_states

#TODO: convert all the parameters to a config object
def main(num_generations=20, num_self_play_games=200, num_training_epochs=3,
         batch_size=128, learning_rate=0.0001, max_moves=500,
         self_play_device="cpu", mcts_batch_size=32,
         mcts_depths=(200, 400, 800, 1200),
         mcts_initial_probs=(0.5, 0.4, 0.1, 0.0),
         mcts_final_probs=(0.1, 0.3, 0.55, 0.05),
         eval_every=1, eval_games_per_color=5, eval_mcts_rounds=100,
         output_root=None):
    """
    self_play_device: "cpu" (default) uses the multiprocessing CPU pool; "mps" or "cuda"
        runs self-play single-process with the batched MCTS agent so the GPU is utilized.
    mcts_batch_size: number of leaves evaluated per network forward pass (GPU path only).
    mcts_depths / *_probs: the Progressive MCTS simulation-count schedule. Lower the depths
        for short experimental runs (e.g. depths=(80, 160)).
    eval_every: run a greedy strength eval (vs RandomAgent + previous model) every N generations
        (0 disables). eval_games_per_color games are played as each color.
    output_root: where to write checkpoints + replay buffer. Defaults to the project dir; set to
        a persistent path (e.g. a Modal Volume mount like "/data") for cloud runs.
    """
    project_root = Path(__file__).resolve().parents[1]
    base_dir = Path(output_root) if output_root is not None else project_root
    checkpoints_dir = base_dir / "model" / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # max_moves must match the game's move cap or the encoder's move-count plane saturates.
    encoder = SevenPlaneEncoder(board_size=11, max_moves=max_moves)
    current_model = DualNetwork(encoder=encoder, learning_rate=learning_rate, batch_size=batch_size)
    persistent_buffer = PersistentExperienceBuffer(max_games=75_000)
    stat_tracker = StatTracker()

    start_generation = 0
    wandb_run_id = None  # wandb requires a STRING run id; None lets it generate one. Overwritten on resume.
    latest_ckpt_path = checkpoints_dir / "last-v5.ckpt"
    replay_buffer_path = base_dir / "data" / "replay_buffer.pkl"

    if latest_ckpt_path.exists():
        print(f"Found checkpoint. Resuming training from: {latest_ckpt_path}")
        full_ckpt = torch.load(latest_ckpt_path, weights_only=False)
        current_model.load_state_dict(full_ckpt['state_dict'])
        start_generation = full_ckpt.get('current_generation', 0)
        wandb_run_id = full_ckpt.get('wandb_run_id')
        print(f"Resuming from generation {start_generation} with run ID {wandb_run_id}")
    else:
        print("No checkpoint found. Starting new training run.")

    # Reload any existing replay buffer regardless of checkpoint, so a crash during training
    # (after self-play has been saved) doesn't throw away the generated games on restart.
    try:
        persistent_buffer.set_states(load_replay_buffer(replay_buffer_path))
        print(f"Loaded {len(persistent_buffer)} states from replay buffer at {replay_buffer_path}.")
    except FileNotFoundError:
        print(f"No replay buffer at {replay_buffer_path}; starting with an empty buffer.")

    mcts_config = ProgressiveMCTSConfigs(
        depths=list(mcts_depths),
        initial_probs=list(mcts_initial_probs),
        final_probs=list(mcts_final_probs),
        total_gens=num_generations
    )

    # CPU snapshot of the model from the previous eval, used as the "previous" opponent.
    previous_eval_state_dict = None

    for generation in range(start_generation, num_generations):
        gc.collect()
        print(f"\n===== STARTING GENERATION {generation + 1}/{num_generations} =====")
        print(f"Current buffer size: {len(persistent_buffer)}")

        print("--- Generating self-play games... ---")
        current_generation = generation + 1
        current_model.eval()
        current_model.cpu()  # Lightning may have left the model on MPS after training;
                             # ensure CPU tensors before pickling state_dict to spawn workers.
        torch.set_grad_enabled(False)

        if self_play_device != "cpu":
            # GPU path: sequential games in this process, batched leaf eval keeps the device busy.
            all_results = _self_play_in_process(current_model, encoder, mcts_config, generation,
                                                max_moves, num_self_play_games,
                                                device=self_play_device,
                                                mcts_batch_size=mcts_batch_size)
            current_model.to("cpu")  # hand a CPU model to Lightning; it relocates as needed
        else:
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
        save_replay_buffer(persistent_buffer, replay_buffer_path)
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
            callbacks=[checkpoint_callback],
            enable_progress_bar=True,
            log_every_n_steps=10
        )

        trainer.fit(current_model, train_dataloaders=train_loader)

        stat_tracker.summarize_generation(current_generation)

        # ---- Greedy strength eval: the real signal, separate from noisy self-play stats ----
        if eval_every > 0 and current_generation % eval_every == 0:
            print("--- Evaluating (greedy) vs RandomAgent and previous model... ---")
            current_model.cpu().eval()
            torch.set_grad_enabled(False)

            rand_res = evaluate_vs_random(current_model, encoder,
                                          n_games_per_color=eval_games_per_color,
                                          model_mcts_rounds=eval_mcts_rounds, max_moves=max_moves)
            print("  " + rand_res.summary())
            eval_metrics = {
                "eval/vs_random_score": rand_res.score,
                "eval/vs_random_black": rand_res.score_as_black,
                "eval/vs_random_white": rand_res.score_as_white,
            }

            if previous_eval_state_dict is not None:
                prev_model = DualNetwork(encoder=encoder)
                prev_model.load_state_dict(previous_eval_state_dict)
                prev_model.cpu().eval()
                prev_res = evaluate_vs_model(current_model, prev_model, encoder,
                                             n_games_per_color=eval_games_per_color,
                                             model_mcts_rounds=eval_mcts_rounds, max_moves=max_moves)
                print("  " + prev_res.summary())
                eval_metrics["eval/vs_previous_score"] = prev_res.score

            wandb_logger.experiment.log(eval_metrics)

            # Snapshot current weights to be the "previous" opponent at the next eval.
            previous_eval_state_dict = {k: v.detach().cpu().clone()
                                        for k, v in current_model.state_dict().items()}
            torch.set_grad_enabled(True)

    print("Training complete.")
    stat_tracker.close()


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)

    # Short experimental run (~1-2h on an 8-core M1): many small generations so learning
    # trends (win-rate-by-color, game length, losses) become visible quickly. Self-play on
    # CPU (multiprocessing gives ~6x throughput); Lightning trains on MPS automatically.
    main(num_generations=15, num_self_play_games=48, num_training_epochs=2,
         batch_size=128, learning_rate=0.0001, max_moves=200,
         self_play_device="cpu",
         mcts_depths=(100, 200), mcts_initial_probs=(0.7, 0.3), mcts_final_probs=(0.3, 0.7))

    # Full run (slower, stronger): bigger schedule + more games.
    # main(num_generations=20, num_self_play_games=200, num_training_epochs=5,
    #      batch_size=128, learning_rate=0.0001, max_moves=400)
