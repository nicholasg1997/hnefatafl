# Hnefatafl AI (AlphaZero-style)

This repository contains an AlphaZero-style self-play system that learns to play the ancient Norse board game Hnefatafl using Monte Carlo Tree Search (MCTS) guided by a neural network.

The implementation uses PyTorch / PyTorch Lightning and multiprocessing for fast self-play.

## Features
- Self-play loop with MCTS (ZeroAgent fast implementation)
- Dual-head CNN policy/value network
- Persistent experience buffer and iterative training
- Weights & Biases logging (requires login)

## Installation
- Python: >= 3.10
- This project uses uv for environment and dependency management (uv.lock is provided).

Setup with uv:

```bash
# install uv if needed: https://docs.astral.sh/uv/getting-started/installation/
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv sync  # installs all dependencies and the project itself
```

Weights & Biases (required for training):

```bash
uv run wandb login
```

## Quick Start: Training
The main training entry point is:

```bash
uv run python hnefatafl/zero/lightningTrainer.py
```

Key notes:
- Uses SevenPlaneEncoder (11x11 default) and ZeroAgent from `hnefatafl/zero/zeroagent_fast.py` with Progressive MCTS sampling.
- Checkpointing: saves PyTorch Lightning checkpoints under `model/checkpoints/` and will resume from `model/checkpoints/last-v4.ckpt` if present.
- Experience replay is persisted via ReplayBufferCheckpoint so training can resume across runs.
- Multiprocessing: automatically uses available CPU cores while leaving a couple free by default.
- Weights & Biases logging via `WandbLogger` (project `hnefatafl-zero`); run IDs are restored from checkpoints when resuming.

Customize hyperparameters by editing the `main(...)` defaults in `lightningTrainer.py` or by importing it:

```python
from hnefatafl.zero.lightningTrainer import main

main(
    num_generations=20,
    num_self_play_games=200,
    num_training_epochs=5,
    batch_size=128,
    learning_rate=0.0001,
    max_moves=400,
)
```


## Project Structure (highlights)
- `hnefatafl/core`: Core game logic and types
- `hnefatafl/encoders/advanced_encoder.py`: SevenPlaneEncoder used by the networks
- `hnefatafl/zero/`:
  - `lightningTrainer.py`: self-play training loop (entry point)
  - `lightning_network.py`: PyTorch LightningModule dual-head network used for training
  - `zeroagent_fast.py`: fast MCTS agent used for training and play
  - `experienceCollector_v2.py`: self-play experience collection and buffers
- `hnefatafl/utils/nnTrainingUtils.py`: simulate_game_simple and helpers
- `hnefatafl/models/blackWhiteNetwork.py`: alternative experimental DualNetwork with split value heads

## Tips
- GPU: Training can run on CPU or GPU; ensure PyTorch detects CUDA if available and configure your environment accordingly.
- Reproducibility: For strict reproducibility, seed numpy/torch and control multiprocessing seeds as needed.
- Logging: Training uses a `WandbLogger` (project `hnefatafl-zero`); you must be logged into Weights & Biases (`wandb login`).

## Acknowledgments
- Inspired by AlphaGo Zero/AlphaZero and the book "Deep Learning and the Game of Go" by Max Pumperla and Kevin Ferguson.
