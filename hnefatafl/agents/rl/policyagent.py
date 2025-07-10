from hnefatafl.agents.agent import Agent
from hnefatafl.encoders.base import get_encoder_by_name
from hnefatafl.utils.nnTrainingUtils import clip_probs
from hnefatafl.core.move import Move
import torch

class PolicyAgent(Agent):
    def __init__(self, model, encoder, collector=None):
        self.model = model
        self.encoder = encoder
        self.collector = collector

    def name(self):
        return 'PolicyAgent'

    def set_collector(self, collector):
        self.collector = collector

    def select_move(self, game_state):
        board_tensor = self.encoder.encode(game_state)
        X = torch.tensor(board_tensor, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(X)

        legal_moves = game_state.get_legal_moves()
        if not legal_moves:
            print("No legal moves available.")
            return None

        legal_moves_indices = [Move.encode(move) for move in legal_moves]
        logit_mask = torch.ones_like(logits[0]) * -1e9
        logit_mask[legal_moves_indices] = 0
        masked_logits = logits + logit_mask
        probs = torch.softmax(masked_logits, dim=-1)
        probs = clip_probs(probs)
        move_index = torch.multinomial(probs, num_samples=1).item()
        move = Move.from_encoded(move_index)
        if self.collector is not None:
            self.collector.record_decision(board_tensor, move)
        return move

    def make_move(self, game_state):
        move = self.select_move(game_state)
        if move is None:
            raise ValueError("No valid move selected.")
        return game_state.apply_move(move)

    def save_model(self, path):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'encoder_name': self.encoder.name(),
            'board_size': self.encoder.board_width,
            }

        torch.save(checkpoint, path)

    @classmethod
    def load_model(cls, path, model_class, device='cpu', inference=False):
        checkpoint = torch.load(path, map_location=device)
        encoder = get_encoder_by_name(checkpoint['encoder_name'], checkpoint['board_size'])
        model = model_class(encoder).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if inference:
            model.eval()
        return cls(model, encoder)

if __name__ == "__main__":
    import torch
    import torch.nn as nn
    from hnefatafl.encoders.one_plane import OnePlaneEncoder
    from hnefatafl.rl.experience import ExperienceCollector
    from hnefatafl.utils.nnTrainingUtils import simulate_game
    from hnefatafl.core.gameTypes import Player
    import h5py



    class SimplePolicyNet(nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder
            board_height, board_width = encoder.board_height, encoder.board_width
            input_channels = encoder.num_planes
            output_dim = encoder.num_moves()

            self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

            self.relu = nn.ReLU()
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(64 * 11 * 11, output_dim)

        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.flatten(x)
            return self.fc(x)  # returns logits


    encoder = OnePlaneEncoder(board_size=11)
    model = SimplePolicyNet(encoder)

    # black is attacker, white is defender

    black_agent = PolicyAgent(model, encoder)
    white_agent = PolicyAgent(model, encoder)
    black_collector = ExperienceCollector()
    white_collector = ExperienceCollector()
    black_agent.set_collector(black_collector)
    white_agent.set_collector(white_collector)

    for i in range(1):
        black_collector.begin_episode()
        white_collector.begin_episode()

        game_winner = simulate_game(black_agent, white_agent)
        print(f"Game {i + 1} winner: {game_winner}")
        if game_winner is None:
            print("Game ended in a draw.")
            black_collector.complete_episode(reward=0)
            white_collector.complete_episode(reward=0)
        elif game_winner == Player.black:
            print("Black wins!")
            black_collector.complete_episode(reward=1)
            white_collector.complete_episode(reward=-1)
        elif game_winner == Player.white:
            print("White wins!")
            black_collector.complete_episode(reward=-1)
            white_collector.complete_episode(reward=1)
        else:
            print("Unexpected game outcome.")

    from pathlib import Path
    import h5py

    # Current file location
    current_file = Path(__file__)

    # Path to the "experience" folder inside "save_data"
    save_dir = current_file.parent.parent.parent.parent / "save_data" / "experience"

    # Ensure it exists
    # save_dir.mkdir(parents=True, exist_ok=True)

    # Full paths to the .h5 files
    black_path = save_dir / "black.h5"
    white_path = save_dir / "white.h5"

    # Save black experience
    with h5py.File(black_path, 'w') as h5file:
        black_collector.to_buffer().serialize(h5file)

    # Save white experience
    with h5py.File(white_path, 'w') as h5file:
        white_collector.to_buffer().serialize(h5file)

















