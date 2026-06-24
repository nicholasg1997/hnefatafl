"""
Run Hnefatafl AlphaZero training on Modal with a CUDA GPU.

ONE-TIME SETUP (on your machine):
    uv add modal
    modal token new                                  # authenticate to your Modal account
    modal secret create wandb WANDB_API_KEY=<key>    # so W&B logging works in the cloud

LAUNCH:
    modal run modal_train.py                         # cheap validation run (6 gens, T4, ~few $)
    modal run modal_train.py --num-generations 40 --num-self-play-games 128   # bigger run

To scale up after validation: bump num_generations, switch gpu="A10G", and use the deeper
mcts schedule noted in train() below.

Outputs (checkpoints + replay buffer) persist in the "hnefatafl-data" Modal Volume, so runs
resume across launches. Inspect/download with:  modal volume ls hnefatafl-data

NOTE: Modal's API evolves; this targets the current idiom (add_local_python_source, Volume.commit).
If a call signature mismatches your installed modal version, it's a quick fix — ping me.
"""
import modal

app = modal.App("hnefatafl-zero")

# CUDA torch is the default wheel on Linux, so a plain pip_install("torch") gets the GPU build.
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.7.1",
        "lightning>=2.5.2",
        "pytorch-lightning>=2.5.2",
        "numba>=0.61.2",
        "wandb>=0.21.1",
        "h5py>=3.14.0",
        "numpy<2.3",
        "tqdm",
    )
    .add_local_python_source("hnefatafl")  # ships the local hnefatafl package into the image
)

volume = modal.Volume.from_name("hnefatafl-data", create_if_missing=True)
DATA_DIR = "/data"


# --- COST-SAFE VALIDATION CONFIG ---
# T4 is the cheapest GPU and trains this small net fine (the GPU is mostly idle during the
# CPU-bound self-play anyway). Keeping 16 vCPUs makes self-play finish faster, which actually
# *lowers* cost because the GPU bills the whole time it's up. A short 6-generation run is just
# a sanity check: confirm Modal works end-to-end and watch eval/vs_random_score for early signal.
# Estimated ~2-4h, roughly a few dollars. Scale up (A10G, more gens) only if the curve is alive.
@app.function(
    image=image,
    gpu="T4",
    cpu=16.0,
    memory=16384,
    timeout=6 * 60 * 60,
    volumes={DATA_DIR: volume},
    secrets=[modal.Secret.from_name("wandb")],
)
def train(
    num_generations: int = 6,
    num_self_play_games: int = 64,
    num_training_epochs: int = 3,
    batch_size: int = 256,
    learning_rate: float = 1e-4,
    max_moves: int = 200,
    self_play_device: str = "cpu",   # CPU multiproc self-play; Lightning still trains on the GPU
    eval_every: int = 2,
):
    import multiprocessing
    import torch
    from hnefatafl.zero.lightningTrainer import main

    # spawn (not fork) for the worker pool: matches the tested path and avoids CUDA-fork issues.
    multiprocessing.set_start_method("spawn", force=True)
    print("CUDA available:", torch.cuda.is_available(),
          "| device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")

    try:
        main(
            num_generations=num_generations,
            num_self_play_games=num_self_play_games,
            num_training_epochs=num_training_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_moves=max_moves,
            self_play_device=self_play_device,
            eval_every=eval_every,
            # Moderate depth for the validation run: deeper than the local experiment (100/200),
            # cheap enough to keep the run short. Bump to (200,400,800) for a full run.
            mcts_depths=(200, 400),
            mcts_initial_probs=(0.6, 0.4),
            mcts_final_probs=(0.3, 0.7),
            output_root=DATA_DIR,
        )
    finally:
        volume.commit()  # persist checkpoints + replay buffer to the Volume


@app.local_entrypoint()
def cli(
    num_generations: int = 6,
    num_self_play_games: int = 64,
    max_moves: int = 200,
    self_play_device: str = "cpu",
    eval_every: int = 2,
):
    train.remote(
        num_generations=num_generations,
        num_self_play_games=num_self_play_games,
        max_moves=max_moves,
        self_play_device=self_play_device,
        eval_every=eval_every,
    )
