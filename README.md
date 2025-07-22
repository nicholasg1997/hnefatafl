Of course. Here is a README file for your project.

# Hnefatafl AI

This repository contains the source code for an AI that learns to play the ancient Norse board game, Hnefatafl. The project implements a sophisticated self-learning agent based on the principles of AlphaGo Zero.

## Inspiration

The architecture and training methodology are heavily inspired by the concepts detailed in the book **"Deep Learning and the Game of Go"** by Max Pumperla and Kevin Ferguson. It applies a combination of Monte Carlo Tree Search (MCTS) and deep neural networks to master the game's complex, asymmetric strategy.

## How It Works

The AI learns entirely through self-play, following a process similar to AlphaZero:

1.  **Self-Play**: The current best neural network plays games against itself. For each move, a Monte Carlo Tree Search (MCTS) is performed to explore the most promising lines of play.
2.  **Experience Collection**: The data from these games—including the board state, the MCTS search probabilities, and the final game winner—is stored in an experience buffer.
3.  **Training**: The neural network is trained on batches of data sampled from the experience buffer. It learns to:
    *   **Predict the MCTS search results** (the policy head).
    *   **Predict the eventual game outcome** from the current position (the value head).
4.  **Iteration**: The newly trained network becomes the new "best" network, and the cycle repeats. Over many generations, the agent's play becomes progressively stronger.

## Core Components

*   `hnefatafl/core`: Contains the core game logic, including board representation, rules for movement and capture, and game state management.
*   `hnefatafl/encoders`: Includes encoders that transform the game state into a numerical format (a tensor) suitable for the neural network.
*   `hnefatafl/zero`: This is the heart of the AI.
    *   `network.py`: Defines the `DualNetwork` (policy and value heads) using PyTorch and PyTorch Lightning.
    *   `zeroagent_v2.py`: Implements the MCTS agent that uses the neural network to guide its search.
    *   `experienceCollector_v2.py`: Manages the collection and storage of game data from self-play.
    *   `training_v2.py`: The main training script that orchestrates the entire self-play and learning loop using multiprocessing.

## How to Use

To begin training a new agent from scratch, you can run the main training script.

### Prerequisites

Ensure you have the required Python packages installed, including:
*   `torch`
*   `pytorch-lightning`
*   `numpy`
*   `tqdm`

### Running the Training

You can start the training process by executing the `training_v2.py` script. You can configure parameters such as the number of games, MCTS rounds, and batch size directly within the script's `main` function.

```bash
python hnefatafl/zero/training_v2.py
```

The script will use multiple CPU cores to generate self-play games in parallel, collect the experience, and then train the neural network. Trained models will be saved periodically.
