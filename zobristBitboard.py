import random
import numpy as np
from gameTypes import Player, Point
from move import Move
import copy
from collections import deque

from boardConfigs import BOARD_CONFIGS

# Piece constants
EMPTY = 0
BLACK_PAWN = 1
WHITE_PAWN = 2
KING = 3


class ZobristHasher:
    def __init__(self, board_size=11):
        self.board_size = board_size
        self.piece_keys = {}
        self.player_key = random.getrandbits(64)

        # Generate random 64-bit keys for each piece type at each position
        for piece_type in [BLACK_PAWN, WHITE_PAWN, KING]:
            self.piece_keys[piece_type] = []
            for row in range(board_size):
                self.piece_keys[piece_type].append([])
                for col in range(board_size):
                    self.piece_keys[piece_type][row].append(random.getrandbits(64))

    def hash_position(self, board_grid, current_player):
        """Generate Zobrist hash for current board position"""
        hash_value = 0

        # Hash piece positions
        for row in range(self.board_size):
            for col in range(self.board_size):
                piece = board_grid[row, col]
                if piece != EMPTY:
                    hash_value ^= self.piece_keys[piece][row][col]

        # Hash current player
        if current_player == Player.white:
            hash_value ^= self.player_key

        return hash_value

    def update_hash(self, current_hash, move, piece_type, current_player):
        """Incrementally update hash after a move"""
        new_hash = current_hash

        # Remove piece from old position
        new_hash ^= self.piece_keys[piece_type][move.from_pos.row][move.from_pos.col]

        # Add piece to new position
        new_hash ^= self.piece_keys[piece_type][move.to_pos.row][move.to_pos.col]

        # Toggle player
        new_hash ^= self.player_key

        return new_hash


class BitBoard:
    def __init__(self, board_size=11):
        self.board_size = board_size
        self.black_pawns = 0
        self.white_pawns = 0
        self.king = 0
        self.all_pieces = 0

        # Precompute useful bitboards
        self.corners = self._create_corners_bitboard()
        self.throne = self._create_throne_bitboard()
        self.edges = self._create_edges_bitboard()

        # Direction masks for sliding moves
        self.direction_masks = self._create_direction_masks()

    def _pos_to_bit(self, row, col):
        """Convert board position to bit position"""
        return row * self.board_size + col

    def _bit_to_pos(self, bit_pos):
        """Convert bit position to board coordinates"""
        return bit_pos // self.board_size, bit_pos % self.board_size

    def _create_corners_bitboard(self):
        """Create bitboard with corner positions set"""
        corners = 0
        corners |= (1 << self._pos_to_bit(0, 0))
        corners |= (1 << self._pos_to_bit(0, self.board_size - 1))
        corners |= (1 << self._pos_to_bit(self.board_size - 1, 0))
        corners |= (1 << self._pos_to_bit(self.board_size - 1, self.board_size - 1))
        return corners

    def _create_throne_bitboard(self):
        """Create bitboard with throne position set"""
        center = self.board_size // 2
        return 1 << self._pos_to_bit(center, center)

    def _create_edges_bitboard(self):
        """Create bitboard with edge positions set"""
        edges = 0
        for i in range(self.board_size):
            edges |= (1 << self._pos_to_bit(0, i))  # Top edge
            edges |= (1 << self._pos_to_bit(self.board_size - 1, i))  # Bottom edge
            edges |= (1 << self._pos_to_bit(i, 0))  # Left edge
            edges |= (1 << self._pos_to_bit(i, self.board_size - 1))  # Right edge
        return edges

    def _create_direction_masks(self):
        """Create masks for each direction from each position"""
        masks = {}
        for row in range(self.board_size):
            for col in range(self.board_size):
                pos = self._pos_to_bit(row, col)
                masks[pos] = {
                    'north': self._create_ray_mask(row, col, -1, 0),
                    'south': self._create_ray_mask(row, col, 1, 0),
                    'east': self._create_ray_mask(row, col, 0, 1),
                    'west': self._create_ray_mask(row, col, 0, -1)
                }
        return masks

    def _create_ray_mask(self, start_row, start_col, dr, dc):
        """Create a mask for a ray in a specific direction"""
        mask = 0
        row, col = start_row + dr, start_col + dc
        while 0 <= row < self.board_size and 0 <= col < self.board_size:
            mask |= (1 << self._pos_to_bit(row, col))
            row += dr
            col += dc
        return mask

    def set_piece(self, row, col, piece_type):
        """Set a piece at the given position"""
        bit_pos = 1 << self._pos_to_bit(row, col)

        # Clear the position first
        self.clear_position(row, col)

        # Set the piece
        if piece_type == BLACK_PAWN:
            self.black_pawns |= bit_pos
        elif piece_type == WHITE_PAWN:
            self.white_pawns |= bit_pos
        elif piece_type == KING:
            self.king |= bit_pos

        self.all_pieces |= bit_pos

    def clear_position(self, row, col):
        """Clear a position"""
        bit_pos = 1 << self._pos_to_bit(row, col)
        inverse_bit = ~bit_pos

        self.black_pawns &= inverse_bit
        self.white_pawns &= inverse_bit
        self.king &= inverse_bit
        self.all_pieces &= inverse_bit

    def get_piece_at(self, row, col):
        """Get piece type at position"""
        bit_pos = 1 << self._pos_to_bit(row, col)

        if self.black_pawns & bit_pos:
            return BLACK_PAWN
        elif self.white_pawns & bit_pos:
            return WHITE_PAWN
        elif self.king & bit_pos:
            return KING
        else:
            return EMPTY

    def move_piece(self, from_row, from_col, to_row, to_col):
        """Move a piece from one position to another"""
        piece_type = self.get_piece_at(from_row, from_col)
        self.clear_position(from_row, from_col)
        self.set_piece(to_row, to_col, piece_type)

    def get_sliding_moves(self, row, col, piece_type):
        """Get all sliding moves for a piece at given position"""
        moves = []
        pos = self._pos_to_bit(row, col)

        for direction, ray_mask in self.direction_masks[pos].items():
            # Find blocking pieces
            blockers = ray_mask & self.all_pieces

            # Generate moves until first blocker
            dr, dc = {'north': (-1, 0), 'south': (1, 0), 'east': (0, 1), 'west': (0, -1)}[direction]

            current_row, current_col = row + dr, col + dc
            while (0 <= current_row < self.board_size and 0 <= current_col < self.board_size):
                if self.get_piece_at(current_row, current_col) != EMPTY:
                    break

                # Check if non-king pieces can move to throne/corners
                to_pos = Point(current_row, current_col)
                if piece_type != KING:
                    bit_pos = 1 << self._pos_to_bit(current_row, current_col)
                    if (bit_pos & self.corners) or (bit_pos & self.throne):
                        current_row += dr
                        current_col += dc
                        continue

                moves.append(Point(current_row, current_col))
                current_row += dr
                current_col += dc

        return moves

    def is_king_captured(self):
        """Check if king is captured using bitboard operations"""
        if not self.king:
            return True

        # Find king position
        king_pos = (self.king).bit_length() - 1
        king_row, king_col = self._bit_to_pos(king_pos)

        # Check all four directions around king
        attacking_pawns = 0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor_row, neighbor_col = king_row + dr, king_col + dc

            if 0 <= neighbor_row < self.board_size and 0 <= neighbor_col < self.board_size:
                if not self._is_hostile_to_king(neighbor_row, neighbor_col):
                    return False
                if self.get_piece_at(neighbor_row, neighbor_col) == BLACK_PAWN:
                    attacking_pawns += 1

        return attacking_pawns >= 3

    def _is_hostile_to_king(self, row, col):
        """Check if position is hostile to king"""
        piece = self.get_piece_at(row, col)
        if piece == BLACK_PAWN:
            return True

        bit_pos = 1 << self._pos_to_bit(row, col)

        # Check corners
        if bit_pos & self.corners:
            return True

        # Check throne
        if bit_pos & self.throne:
            throne_piece = self.get_piece_at(self.board_size // 2, self.board_size // 2)
            return throne_piece == EMPTY

        return False

    def copy(self):
        """Create a deep copy of the bitboard"""
        new_board = BitBoard(self.board_size)
        new_board.black_pawns = self.black_pawns
        new_board.white_pawns = self.white_pawns
        new_board.king = self.king
        new_board.all_pieces = self.all_pieces
        return new_board

    def from_numpy_grid(self, grid):
        """Initialize bitboard from numpy grid"""
        self.black_pawns = 0
        self.white_pawns = 0
        self.king = 0
        self.all_pieces = 0

        for row in range(self.board_size):
            for col in range(self.board_size):
                piece = grid[row, col]
                if piece != EMPTY:
                    self.set_piece(row, col, piece)



class Board:

    def __init__(self, board=11):
        self.board_config = BOARD_CONFIGS[board]
        self.size = self.board_config['size']
        self.grid = np.zeros((self.size, self.size), dtype=int)

        # Add bitboard and zobrist hash support
        self.bitboard = BitBoard(self.size)
        self.zobrist_hasher = ZobristHasher(self.size)
        self.position_hash = 0

        center = self.size // 2
        self.throne = Point(center, center)
        self.corners = [
            Point(0, 0),
            Point(0, self.size - 1),
            Point(self.size - 1, 0),
            Point(self.size - 1, self.size - 1)
        ]
        self.set_up_board()

    def reset(self):
        self.set_up_board()

    def set_up_board(self):
        self.current_player = Player.black
        self.grid = np.copy(self.board_config['board'])

        # Update bitboard and hash
        self.bitboard.from_numpy_grid(self.grid)
        self.position_hash = self.zobrist_hasher.hash_position(self.grid, self.current_player)

    def is_on_board(self, point):
        return 0 <= point.row < self.size and 0 <= point.col < self.size

    def get_pawn_at(self, point):
        if not self.is_on_board(point):
            return None
        # Use bitboard for faster lookup
        return self.bitboard.get_piece_at(point.row, point.col)

    def move_pawn(self, move):
        piece = self.get_pawn_at(move.from_pos)

        # Update numpy grid
        self.grid[move.from_pos.row, move.from_pos.col] = EMPTY
        self.grid[move.to_pos.row, move.to_pos.col] = piece

        # Update bitboard
        self.bitboard.move_piece(move.from_pos.row, move.from_pos.col,
                                 move.to_pos.row, move.to_pos.col)

        # Update hash incrementally
        self.position_hash = self.zobrist_hasher.update_hash(
            self.position_hash, move, piece, self.current_player)

    def get_sliding_moves_fast(self, point):
        """Fast move generation using bitboard"""
        piece = self.get_pawn_at(point)
        if piece == EMPTY:
            return []

        return self.bitboard.get_sliding_moves(point.row, point.col, piece)

    def is_king_captured_fast(self):
        """Fast king capture detection using bitboard"""
        return self.bitboard.is_king_captured()

    def find_king(self):
        """Find the position of the king using bitboard"""
        king_pos = self.bitboard.king
        if king_pos == 0:
            return None
        # Get the position of the king from the bitboard
        king_index = (king_pos).bit_length() - 1
        return Point(*self.bitboard._bit_to_pos(king_index))

    def copy(self):
        """Create a copy of the board with bitboard support"""
        new_board = Board(self.size)
        new_board.grid = np.copy(self.grid)
        new_board.bitboard = self.bitboard.copy()
        new_board.position_hash = self.position_hash
        new_board.current_player = self.current_player
        return new_board

    def __str__(self) -> str:
        symbols = {0: '.', 1: 'B', 2: 'W', 3: 'K'}
        rendered = []
        for row in self.grid:
            rendered.append(' '.join(symbols[cell] for cell in row))
        return '\n'.join(rendered)

    def __hash__(self):
        """Use Zobrist hash for fast position comparison"""
        return self.position_hash

    def __eq__(self, other):
        """Fast equality check using Zobrist hash"""
        if not isinstance(other, Board):
            return False
        return self.position_hash == other.position_hash

class GameState:
    def __init__(self, board, next_player, previous, move):
        self.board = board
        self.next_player = next_player
        self.previous = previous
        self.last_move = move
        self.winner = None

    @classmethod
    def new_game(cls, board_size=11):
        return cls(Board(board_size), Player.black, None, None)

    def apply_move(self, move):
        new_board = Board(self.board.size)
        new_board.grid = copy.deepcopy(self.board.grid)
        # ensure that the pawn being moved is the correct player's pawn
        moving_pawn = self.board.get_pawn_at(move.from_pos)
        if self.next_player == Player.white:
            if moving_pawn not in [WHITE_PAWN, KING]:
                raise ValueError("Invalid move: not a white pawn or king")
        else:
            if moving_pawn != BLACK_PAWN:
                raise ValueError("Invalid move: not a black pawn")

        new_board.move_pawn(move)

        next_state = GameState(new_board, self.next_player.other, self, move)
        next_state._check_for_capture(move, self.next_player)
        next_state.is_over()
        return next_state

    def is_over(self):
        if self.winner is not None:
            return True

        if self.board.is_king_captured_fast():
            self.winner = Player.black
            return True

        king_pos = self.board.find_king()
        if king_pos & self.board.corners:
            self.winner = Player.white
            return True

        # TODO: Implement fortress checks and if all pawns have been captured checks

        return False

    def get_legal_moves(self):
        legal_moves = []

        # Determine which pieces belong to the current player
        if self.next_player == Player.white:
            my_pawns = self.board.bitboard.white_pawns
            my_king = self.board.bitboard.king
            pieces_to_move = my_pawns | my_king
        else:  # Player is black
            pieces_to_move = self.board.bitboard.black_pawns

        # Iterate through only the squares occupied by our pieces
        while pieces_to_move > 0:
            # Get the position of the next piece
            piece_pos_index = (pieces_to_move).bit_length() - 1
            row, col = self.board.bitboard._bit_to_pos(piece_pos_index)
            from_pos = Point(row, col)

            # Get all valid sliding moves for this piece using the fast bitboard method
            # The get_sliding_moves method already handles all the logic for blockers and restricted squares.
            end_positions = self.board.get_sliding_moves_fast(from_pos)

            for to_pos in end_positions:
                legal_moves.append(Move(from_pos, to_pos))

            # Remove this piece from the bitboard so we can move to the next one
            pieces_to_move &= ~(1 << piece_pos_index)

        return legal_moves

    def will_capture(self, opposite_point, player, capture_point):
        if self.is_hostile(opposite_point, player):
            self.capture(capture_point)


    def _check_for_capture(self, move, player):
        opponent_pawn = player.other
        # Check if the new point is adjacent to an opponent's pawn
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Up, Down, Left, Right
            neighbor = Point(move.to_pos.row + dr, move.to_pos.col + dc)
            opposite = Point(move.to_pos.row + 2 * dr, move.to_pos.col + 2 * dc)

            if self.board.is_on_board(neighbor) and self.board.grid[neighbor.row, neighbor.col] == opponent_pawn.value:
                self.will_capture(opposite, player, neighbor)

    def capture(self, point):
        self.board.grid[point.row, point.col] = EMPTY

    def is_hostile(self, point, player):
        # Check if the point is occupied by an opponent's pawn or is a corner point or center point
        if not self.board.is_on_board(point):
            return False
        pawn = self.board.get_pawn_at(point)
        if pawn == player.value:
            return True
        if pawn == KING and player == Player.black:
            # King is only hostile to black pawns
            return True


        if point in self.board.corners:
            return True

        if point == self.board.throne and (self.board.get_pawn_at(self.board.throne) == EMPTY
                                           or self.board.get_pawn_at(self.board.throne) == player.other.value):
            return True

        if pawn == EMPTY:
            return False

        return False

    def _is_king_captured(self, king_pos):
        # Check if the king is captured by checking if it is surrounded by hostile pawns
        attacking_pawns = 0
        for neighbor in king_pos.neighbors():
            if not self.is_hostile(neighbor, Player.black):
                return False
            if self.board.get_pawn_at(neighbor) == BLACK_PAWN:
                attacking_pawns += 1

        return attacking_pawns >= 3

    def find_king(self):
        # Find the position of the king on the board
        for row in range(self.board.size):
            for col in range(self.board.size):
                if self.board.grid[row, col] == KING:
                    return Point(row, col)
        return None

    def _is_fortress(self):
        # Determine if white is stuck in a fortress by doing a bfs from the corners of the boards. If none of the white pawns are in the search space, then it is a fortress
        size = self.board.size
        d = deque([Point(0, 0), Point(0, size - 1), Point(size - 1, 0), Point(size - 1, size - 1)])
        visited = set()

        while d:
            point = d.popleft()
            if point in visited or not self.board.is_on_board(point):
                continue
            visited.add(point)

            pawn = self.board.get_pawn_at(point)

            # Check if the point is occupied by a white pawn
            if pawn == WHITE_PAWN or pawn == KING:
                return False

            if pawn == BLACK_PAWN:
                continue

            # Add neighbors to the queue
            for neighbor in point.neighbors():
                if neighbor not in visited and self.board.is_on_board(neighbor):
                    d.append(neighbor)
        # If we reach here, it means all available points are either empty or occupied by black pawns
        return True

    def _is_shield_wall(self):
        # TODO: check if king can reach the edge of the map and if a black pawn can reach the king,
        #  if it can reach the edge of the board and no black pawns can get to it,
        #  then it is an exit for and white wins the game
        king_pos = self.find_king()
        pass



    def _is_exit_fort(self):
        pass


if __name__ == "__main__":
    board = Board()
    print(board)
    print(f"Position hash: {board.position_hash}")
