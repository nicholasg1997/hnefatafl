import numpy as np
from collections import deque
from hnefatafl.encoders.base import Encoder
from hnefatafl.core.gameTypes import Player, Point
from hnefatafl.core.move import Move

EMPTY = 0
BLACK_PAWN = 1
WHITE_PAWN = 2
KING = 3


class SevenPlaneEncoder(Encoder):
    def __init__(self, board_size, max_moves=200, history_length=7):
        """
        Initialize the encoder with configurable parameters.

        Args:
            board_size: Size of the board (assumes square board)
            max_moves: Maximum number of moves to track
            history_length: Number of previous board states to remember
        """
        if isinstance(board_size, int):
            self.board_size = (board_size, board_size)
        else:
            self.board_size = board_size
        self.board_width, self.board_height = self.board_size

        self.max_moves = max_moves
        self.history_length = history_length

        # Calculate number of channels
        # Base channels: king, corners/throne, black_pawns, white_pawns, turn, edges, move_count
        self.base_channels = 7
        # Historical channels: (black_pawns, white_pawns, king) for each historical state
        self.history_channels = 3 * history_length
        self.total_channels = self.base_channels + self.history_channels

        # Pre-compute static channels
        self._corners_throne_channel = self._create_corners_throne_channel()
        self._edges_channel = self._create_edges_channel()

        # Move space dimensions
        self.num_rows = self.board_size[0]
        self.num_cols = self.board_size[1]
        self.num_directions = 4  # up, down, left, right
        self.max_distance = 10  # maximum distance in any direction
        self.num_planes = self.total_channels

        # Total number of possible moves
        self.move_space_size = self.num_rows * self.num_cols * self.num_directions * self.max_distance

    def name(self):
        return f'seven_plane_{self.board_size[0]}x{self.board_size[1]}'

    def _create_corners_throne_channel(self):
        """Create channel with corners and throne marked as 1"""
        channel = np.zeros(self.board_size)
        size = self.board_size[0]

        # Mark corners
        channel[0, 0] = 1
        channel[0, size - 1] = 1
        channel[size - 1, 0] = 1
        channel[size - 1, size - 1] = 1

        # Mark throne (center)
        center = size // 2
        channel[center, center] = 1

        return channel

    def _create_edges_channel(self):
        """Create channel with all edges marked as 1"""
        channel = np.zeros(self.board_size)
        size = self.board_size[0]

        # Mark all edges
        channel[0, :] = 1  # Top edge
        channel[size - 1, :] = 1  # Bottom edge
        channel[:, 0] = 1  # Left edge
        channel[:, size - 1] = 1  # Right edge

        return channel

    def _create_move_count_channel(self, move_count):
        """Create channel representing move count"""
        channel = np.zeros(self.board_size)
        total_positions = self.board_size[0] * self.board_size[1]

        # Clamp move count to max_moves
        clamped_moves = min(move_count, self.max_moves)

        # Set first n positions to 1, where n is the move count
        if clamped_moves > 0:
            flat_channel = channel.flatten()
            flat_channel[:clamped_moves] = 1
            channel = flat_channel.reshape(self.board_size)

        return channel

    def _get_move_count(self, game_state):
        """Count moves by traversing game history"""
        count = 0
        current = game_state
        while current.previous is not None:
            count += 1
            current = current.previous
        return count

    def _get_board_history(self, game_state):
        """Get last N board states"""
        history = deque(maxlen=self.history_length)
        current = game_state

        # Collect history (most recent first)
        while current is not None and len(history) < self.history_length:
            history.appendleft(current.board.grid)
            current = current.previous

        # Pad with empty boards if needed
        while len(history) < self.history_length:
            history.appendleft(np.zeros(self.board_size, dtype=int))

        return list(history)

    def encode(self, game_state):
        """
        Encode game state into multi-channel representation.

        Returns:
            numpy array of shape (channels, board_height, board_width)
        """
        board_grid = game_state.board.grid
        encoded = np.zeros((self.total_channels, self.board_size[0], self.board_size[1]))

        channel_idx = 0

        # Channel 0: King position
        king_channel = np.zeros(self.board_size)
        king_positions = np.where(board_grid == KING)
        if len(king_positions[0]) > 0:
            king_channel[king_positions] = 1
        encoded[channel_idx] = king_channel
        channel_idx += 1

        # Channel 1: Corners and throne
        encoded[channel_idx] = self._corners_throne_channel
        channel_idx += 1

        # Channel 2: Black pawns
        black_pawn_channel = np.zeros(self.board_size)
        black_positions = np.where(board_grid == BLACK_PAWN)
        if len(black_positions[0]) > 0:
            black_pawn_channel[black_positions] = 1
        encoded[channel_idx] = black_pawn_channel
        channel_idx += 1

        # Channel 3: White pawns
        white_pawn_channel = np.zeros(self.board_size)
        white_positions = np.where(board_grid == WHITE_PAWN)
        if len(white_positions[0]) > 0:
            white_pawn_channel[white_positions] = 1
        encoded[channel_idx] = white_pawn_channel
        channel_idx += 1

        # Channel 4: Current player turn
        turn_channel = np.zeros(self.board_size)
        if game_state.next_player == Player.white:
            turn_channel.fill(1)
        encoded[channel_idx] = turn_channel
        channel_idx += 1

        # Channel 5: Board edges (all ones)
        encoded[channel_idx] = self._edges_channel
        channel_idx += 1

        # Channel 6: Move count
        move_count = self._get_move_count(game_state)
        encoded[channel_idx] = self._create_move_count_channel(move_count)
        channel_idx += 1

        # Historical channels
        board_history = self._get_board_history(game_state)

        for hist_board in board_history:
            # Black pawns in this historical state
            black_hist_channel = np.zeros(self.board_size)
            black_hist_positions = np.where(hist_board == BLACK_PAWN)
            if len(black_hist_positions[0]) > 0:
                black_hist_channel[black_hist_positions] = 1
            encoded[channel_idx] = black_hist_channel
            channel_idx += 1

            # White pawns in this historical state
            white_hist_channel = np.zeros(self.board_size)
            white_hist_positions = np.where(hist_board == WHITE_PAWN)
            if len(white_hist_positions[0]) > 0:
                white_hist_channel[white_hist_positions] = 1
            encoded[channel_idx] = white_hist_channel
            channel_idx += 1

            # King in this historical state
            king_hist_channel = np.zeros(self.board_size)
            king_hist_positions = np.where(hist_board == KING)
            if len(king_hist_positions[0]) > 0:
                king_hist_channel[king_hist_positions] = 1
            encoded[channel_idx] = king_hist_channel
            channel_idx += 1

        return encoded

    def encode_move(self, move):
        """
        Encode a Move object into a flat index for the neural network output.

        Args:
            move: Move object

        Returns:
            int: Flat index representing the move
        """
        return move.encode(board_size=self.board_size[0])

    def decode_move_index(self, move_index):
        """
        Decode a flat move index back into a Move object.

        Args:
            move_index: Flat index from neural network output

        Returns:
            Move: Move object
        """
        return Move.from_encoded(move_index, self.board_size[0])

    def encode_point(self, point):
        """Encode a point to a flat index"""
        return point.row * self.board_size[1] + point.col

    def decode_point_index(self, index):
        """Decode a flat index to a Point"""
        row = index // self.board_size[1]
        col = index % self.board_size[1]
        return Point(row, col)

    def num_points(self):
        """Total number of points on the board"""
        return self.board_size[0] * self.board_size[1]

    def num_moves(self):
        """Total number of possible moves"""
        return self.move_space_size

    def create_move_probabilities(self, legal_moves):
        """
        Create a probability distribution over all possible moves.

        Args:
            legal_moves: List of Move objects that are legal

        Returns:
            numpy array of shape (move_space_size,) with probabilities
        """
        probs = np.zeros(self.move_space_size)
        if legal_moves:
            # Uniform distribution over legal moves
            prob_per_move = 1.0 / len(legal_moves)
            for move in legal_moves:
                move_index = self.encode_move(move)
                probs[move_index] = prob_per_move
        return probs

    def filter_legal_moves(self, move_probabilities, legal_moves):
        """
        Filter move probabilities to only include legal moves.

        Args:
            move_probabilities: numpy array of shape (move_space_size,)
            legal_moves: List of Move objects that are legal

        Returns:
            numpy array of shape (move_space_size,) with only legal moves having non-zero probability
        """
        filtered_probs = np.zeros(self.move_space_size)

        # Create mask for legal moves
        legal_indices = [self.encode_move(move) for move in legal_moves]

        # Copy probabilities for legal moves only
        for idx in legal_indices:
            filtered_probs[idx] = move_probabilities[idx]

        # Renormalize if needed
        total = np.sum(filtered_probs)
        if total > 0:
            filtered_probs /= total

        return filtered_probs

    def sample_move(self, move_probabilities, legal_moves):
        """
        Sample a move from the probability distribution.

        Args:
            move_probabilities: numpy array of shape (move_space_size,)
            legal_moves: List of Move objects that are legal

        Returns:
            Move: Sampled move
        """
        # Filter to only legal moves
        filtered_probs = self.filter_legal_moves(move_probabilities, legal_moves)

        # Sample from the distribution
        if np.sum(filtered_probs) > 0:
            move_index = np.random.choice(self.move_space_size, p=filtered_probs)
            return self.decode_move_index(move_index)
        else:
            # Fallback to random legal move
            return np.random.choice(legal_moves)

    def get_best_move(self, move_probabilities, legal_moves):
        """
        Get the move with highest probability among legal moves.

        Args:
            move_probabilities: numpy array of shape (move_space_size,)
            legal_moves: List of Move objects that are legal

        Returns:
            Move: Best move according to probabilities
        """
        if not legal_moves:
            return None

        best_move = None
        best_prob = -1

        for move in legal_moves:
            move_index = self.encode_move(move)
            prob = move_probabilities[move_index]
            if prob > best_prob:
                best_prob = prob
                best_move = move

        return best_move

    def get_shape(self):
        """Return the shape of the encoded representation"""
        return (self.total_channels, self.board_size[0], self.board_size[1])

    def move_space_shape(self):
        """Return the shape of the move space"""
        return (self.move_space_size,)

    def get_channel_info(self):
        """Get information about what each channel represents"""
        info = {
            0: "King position",
            1: "Corners and throne",
            2: "Black pawns (current)",
            3: "White pawns (current)",
            4: "Current player turn (1=white, 0=black)",
            5: "Board edges (all ones)",
            6: "Move count representation"
        }

        # Add historical channel info
        for i in range(self.history_length):
            base_idx = 7 + i * 3
            info[base_idx] = f"Black pawns (history -{i + 1})"
            info[base_idx + 1] = f"White pawns (history -{i + 1})"
            info[base_idx + 2] = f"King position (history -{i + 1})"

        return info


def create(board_size, max_moves=200, history_length=7):
    """Factory function to create encoder instance"""
    return SevenPlaneEncoder(board_size, max_moves, history_length)
