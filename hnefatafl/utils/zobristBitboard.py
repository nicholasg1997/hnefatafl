import random
import numpy as np
from hnefatafl.core.gameTypes import Player, Point
from hnefatafl.core.move import Move
import copy
from collections import deque

from hnefatafl.core.boardConfigs import BOARD_CONFIGS

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
        """Create masks for each direction (up, down, left, right)"""
        masks = {}
        
        # For each position on the board, create ray masks in all 4 directions
        for row in range(self.board_size):
            for col in range(self.board_size):
                pos = self._pos_to_bit(row, col)
                
                # Up direction
                masks[(pos, 0)] = self._create_ray_mask(row, col, -1, 0)
                # Down direction
                masks[(pos, 1)] = self._create_ray_mask(row, col, 1, 0)
                # Left direction
                masks[(pos, 2)] = self._create_ray_mask(row, col, 0, -1)
                # Right direction
                masks[(pos, 3)] = self._create_ray_mask(row, col, 0, 1)
                
        return masks

    def _create_ray_mask(self, start_row, start_col, dr, dc):
        """Create a bitboard mask for a ray in a given direction"""
        mask = 0
        row, col = start_row + dr, start_col + dc
        
        while 0 <= row < self.board_size and 0 <= col < self.board_size:
            mask |= (1 << self._pos_to_bit(row, col))
            row += dr
            col += dc
            
        return mask

    def set_piece(self, row, col, piece_type):
        """Set a piece on the board"""
        bit_pos = 1 << self._pos_to_bit(row, col)
        
        # Clear any existing piece at this position
        self.black_pawns &= ~bit_pos
        self.white_pawns &= ~bit_pos
        self.king &= ~bit_pos
        
        # Set the new piece
        if piece_type == BLACK_PAWN:
            self.black_pawns |= bit_pos
        elif piece_type == WHITE_PAWN:
            self.white_pawns |= bit_pos
        elif piece_type == KING:
            self.king |= bit_pos
            
        # Update all pieces bitboard
        self.all_pieces = self.black_pawns | self.white_pawns | self.king

    def clear_position(self, row, col):
        """Remove any piece from a position"""
        bit_pos = 1 << self._pos_to_bit(row, col)
        
        # Clear any piece at this position
        self.black_pawns &= ~bit_pos
        self.white_pawns &= ~bit_pos
        self.king &= ~bit_pos
        
        # Update all pieces bitboard
        self.all_pieces = self.black_pawns | self.white_pawns | self.king

    def get_piece_at(self, row, col):
        """Get the piece type at a given position"""
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
        """Get all possible sliding moves for a piece"""
        moves = []
        pos = self._pos_to_bit(row, col)
        
        # Check all four directions
        for direction in range(4):
            # Get the ray mask for this position and direction
            ray_mask = self.direction_masks.get((pos, direction), 0)
            
            # Find the first blocking piece in this direction
            blocking_mask = ray_mask & self.all_pieces
            
            # If there's no blocking piece, we can move to any square in this direction
            if blocking_mask == 0:
                valid_moves_mask = ray_mask
            else:
                # Find the closest blocking piece
                # This is a bit complex in pure Python without specialized bit operations
                # We'll iterate through the ray and stop at the first blocking piece
                valid_moves_mask = 0
                dr, dc = [(0, -1), (0, 1), (-1, 0), (1, 0)][direction]
                r, c = row + dr, col + dc
                
                while 0 <= r < self.board_size and 0 <= c < self.board_size:
                    bit = 1 << self._pos_to_bit(r, c)
                    if self.all_pieces & bit:
                        break
                    valid_moves_mask |= bit
                    r += dr
                    c += dc
            
            # For non-king pieces, exclude corners and throne
            if piece_type != KING:
                valid_moves_mask &= ~(self.corners | self.throne)
                
            # Convert valid moves mask to list of moves
            move_mask = valid_moves_mask
            while move_mask:
                # Find the lowest set bit
                move_bit = move_mask & -move_mask
                move_mask &= ~move_bit
                
                # Convert bit position to board coordinates
                to_row, to_col = self._bit_to_pos(move_bit.bit_length() - 1)
                moves.append(Move(Point(row, col), Point(to_row, to_col)))
                
        return moves

    def is_king_captured(self):
        """Check if the king is captured"""
        if self.king == 0:
            return True  # King is already captured
            
        # Find king position
        king_pos = self.king.bit_length() - 1
        king_row, king_col = self._bit_to_pos(king_pos)
        
        # If king is on throne, it needs to be surrounded on all 4 sides
        if self.king & self.throne:
            # Check all 4 adjacent positions
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                r, c = king_row + dr, king_col + dc
                if not (0 <= r < self.board_size and 0 <= c < self.board_size):
                    return False  # King is at edge, can't be captured
                if not self._is_hostile_to_king(r, c):
                    return False  # Not surrounded
            return True  # King is surrounded on all 4 sides
            
        # If king is adjacent to throne, it needs 3 hostile pieces
        if king_row == self.board_size // 2 and abs(king_col - self.board_size // 2) == 1 or \
           king_col == self.board_size // 2 and abs(king_row - self.board_size // 2) == 1:
            hostile_count = 0
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                r, c = king_row + dr, king_col + dc
                if 0 <= r < self.board_size and 0 <= c < self.board_size and self._is_hostile_to_king(r, c):
                    hostile_count += 1
            return hostile_count >= 3
            
        # Otherwise, king needs to be surrounded on all 4 sides
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            r, c = king_row + dr, king_col + dc
            if not (0 <= r < self.board_size and 0 <= c < self.board_size) or not self._is_hostile_to_king(r, c):
                return False
        return True

    def _is_hostile_to_king(self, row, col):
        """Check if a position is hostile to the king (black pawn, corner, or edge)"""
        bit_pos = 1 << self._pos_to_bit(row, col)
        
        # Black pawns are hostile
        if self.black_pawns & bit_pos:
            return True
            
        # Corners are hostile
        if self.corners & bit_pos:
            return True
            
        # Throne is hostile if empty
        if self.throne & bit_pos and not (self.all_pieces & bit_pos):
            return True
            
        return False

    def copy(self):
        """Create a deep copy of the bitboard"""
        new_board = BitBoard(self.board_size)
        new_board.black_pawns = self.black_pawns
        new_board.white_pawns = self.white_pawns
        new_board.king = self.king
        new_board.all_pieces = self.all_pieces
        
        # No need to copy the precomputed bitboards as they're the same for all boards of the same size
        
        return new_board

    def from_numpy_grid(self, grid):
        """Initialize bitboard from a numpy grid"""
        self.black_pawns = 0
        self.white_pawns = 0
        self.king = 0
        
        for row in range(self.board_size):
            for col in range(self.board_size):
                piece = grid[row, col]
                if piece == BLACK_PAWN:
                    self.black_pawns |= (1 << self._pos_to_bit(row, col))
                elif piece == WHITE_PAWN:
                    self.white_pawns |= (1 << self._pos_to_bit(row, col))
                elif piece == KING:
                    self.king |= (1 << self._pos_to_bit(row, col))
                    
        self.all_pieces = self.black_pawns | self.white_pawns | self.king
        return self


class Board:
    """
    Board representation using bitboards for efficient operations.
    """
    def __init__(self, board=11):
        self.board_config = BOARD_CONFIGS[board]
        self.size = self.board_config['size']
        self.grid = np.zeros((self.size, self.size), dtype=int)
        
        # Initialize bitboard
        self.bitboard = BitBoard(self.size)
        
        # Set up throne and corners
        center = self.size // 2
        self.throne = Point(center, center)
        self.corners = [
            Point(0, 0),
            Point(0, self.size - 1),
            Point(self.size - 1, 0),
            Point(self.size - 1, self.size - 1)
        ]
        
        # Initialize Zobrist hasher
        self.hasher = ZobristHasher(self.size)
        
        self.set_up_board()
        
        # Calculate initial hash
        self.hash_value = self.hasher.hash_position(self.grid, Player.black)

    def reset(self):
        self.set_up_board()

    def set_up_board(self):
        self.current_player = Player.black
        self.grid = np.copy(self.board_config['board'])
        
        # Update bitboard
        self.bitboard.from_numpy_grid(self.grid)
        
        # Recalculate hash
        self.hash_value = self.hasher.hash_position(self.grid, self.current_player)

    def is_on_board(self, point):
        return 0 <= point.row < self.size and 0 <= point.col < self.size

    def get_pawn_at(self, point):
        if not self.is_on_board(point):
            return None
        return self.grid[point.row, point.col]

    def move_pawn(self, move):
        piece = self.get_pawn_at(move.from_pos)
        
        # Update grid
        self.grid[move.from_pos.row, move.from_pos.col] = EMPTY
        self.grid[move.to_pos.row, move.to_pos.col] = piece
        
        # Update bitboard
        self.bitboard.move_piece(move.from_pos.row, move.from_pos.col, move.to_pos.row, move.to_pos.col)
        
        # Update hash
        self.hash_value = self.hasher.update_hash(self.hash_value, move, piece, self.current_player)
        
        # Toggle player
        self.current_player = self.current_player.other

    def get_sliding_moves_fast(self, point):
        """Get all sliding moves for a piece using bitboard operations"""
        piece_type = self.get_pawn_at(point)
        if piece_type == EMPTY:
            return []
        return self.bitboard.get_sliding_moves(point.row, point.col, piece_type)

    def is_king_captured_fast(self):
        """Check if the king is captured using bitboard operations"""
        return self.bitboard.is_king_captured()

    def find_king(self):
        """Find the king's position"""
        if self.bitboard.king == 0:
            return None
            
        king_pos = self.bitboard.king.bit_length() - 1
        row, col = self.bitboard._bit_to_pos(king_pos)
        return Point(row, col)

    def copy(self):
        """Create a deep copy of the board"""
        new_board = Board(11)  # Create with default size
        new_board.size = self.size
        new_board.grid = np.copy(self.grid)
        new_board.bitboard = self.bitboard.copy()
        new_board.hash_value = self.hash_value
        new_board.current_player = self.current_player
        return new_board

    def __str__(self) -> str:
        symbols = {0: '.', 1: 'B', 2: 'W', 3: 'K'}
        rendered = []
        for row in self.grid:
            rendered.append(' '.join(symbols[cell] for cell in row))
        return '\n'.join(rendered)

    def __hash__(self):
        """Use Zobrist hash as the board hash"""
        return self.hash_value

    def __eq__(self, other):
        """Check if two boards are equal"""
        return isinstance(other, Board) and np.array_equal(self.grid, other.grid)


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
        new_board = self.board.copy()
        
        # Ensure that the pawn being moved is the correct player's pawn
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
            
        king_pos = self.board.find_king()
        
        # Check if king is captured
        if king_pos is None or self._is_king_captured(king_pos):
            self.winner = Player.black
            return True
            
        # Check if king reached a corner
        if king_pos in self.board.corners:
            self.winner = Player.white
            return True
            
        # Check for fortress
        if self._is_fortress():
            self.winner = Player.black
            return True
            
        # Check if all pieces of one side are captured
        black_count = np.sum(self.board.grid == BLACK_PAWN)
        white_count = np.sum(self.board.grid == WHITE_PAWN)
        king_count = np.sum(self.board.grid == KING)
        
        if white_count == 0 and king_count == 0:
            self.winner = Player.black
            return True
        if black_count == 0:
            self.winner = Player.white
            return True
            
        return False

    def get_legal_moves(self):
        """
        Calculate all legal moves for the current player using bitboard operations.
        """
        legal_moves = []
        my_pawns = []
        
        # Find all pawns of the current player
        if self.next_player == Player.white:
            # White pawns and king
            for row in range(self.board.size):
                for col in range(self.board.size):
                    if self.board.grid[row, col] in [WHITE_PAWN, KING]:
                        my_pawns.append(Point(row, col))
        else:
            # Black pawns
            for row in range(self.board.size):
                for col in range(self.board.size):
                    if self.board.grid[row, col] == BLACK_PAWN:
                        my_pawns.append(Point(row, col))
        
        # Get all sliding moves for each pawn
        for pawn in my_pawns:
            legal_moves.extend(self.board.get_sliding_moves_fast(pawn))
            
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
        
        # Update bitboard
        self.board.bitboard.clear_position(point.row, point.col)

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
        # Use the optimized bitboard method
        return self.board.is_king_captured_fast()

    def find_king(self):
        # Use the optimized bitboard method
        return self.board.find_king()

    def _is_fortress(self):
        # Determine if white is stuck in a fortress by doing a BFS from the corners
        size = self.board.size
        d = deque([Point(0, 0), Point(0, size - 1), Point(size - 1, 0), Point(size - 1, size - 1)])
        visited = set()
        
        while d:
            point = d.popleft()
            if point in visited or not self.board.is_on_board(point):
                continue
            visited.add(point)
            
            pawn = self.board.get_pawn_at(point)
            
            # Check if the point is occupied by a white pawn or king
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
        # TODO: check if king can reach the edge of the map and if a black pawn can reach the king
        king_pos = self.find_king()
        pass

    def _is_exit_fort(self):
        pass