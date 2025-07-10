from collections import deque
import numpy as np
import random
from collections import Counter

from hnefatafl.core.gameTypes import Player, Point
from hnefatafl.core.board import Board
from hnefatafl.core.move import Move

EMPTY = 0
WHITE_PAWN = 2
BLACK_PAWN = 1
KING = 3


class GameState:
    def __init__(self, board, next_player, previous, move):
        self.board = board
        self.next_player = next_player
        self.previous = previous
        self.last_move = move
        self.winner = None
        # Caching for expensive operations
        self._legal_moves_cache = None
        self._is_over_cache = None
        self._king_pos_cache = None

    @classmethod
    def new_game(cls, board_size=11):
        return cls(Board(board_size), Player.black, None, None)

    def apply_move(self, move):
        # Create new board more efficiently
        new_board = Board(self.board.size)
        new_board.grid = self.board.grid.copy()  # Use copy() instead of deepcopy
        new_board.corners = self.board.corners
        new_board.throne = self.board.throne

        # Validate move (keep for safety)
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
        if self._is_over_cache is not None:
            return self._is_over_cache

        if self.winner is not None:
            self._is_over_cache = True
            return True

        king_pos = self.find_king()
        if king_pos is None or self._is_king_captured(king_pos):
            self.winner = Player.black
            self._is_over_cache = True
            return True

        # Check if king reached corner
        size = self.board.size
        corners = [Point(0, 0), Point(0, size - 1), Point(size - 1, 0), Point(size - 1, size - 1)]
        if king_pos in corners:
            self.winner = Player.white
            self._is_over_cache = True
            return True

        # Use more efficient piece counting
        white_pieces = np.sum((self.board.grid == WHITE_PAWN) | (self.board.grid == KING))
        black_pieces = np.sum(self.board.grid == BLACK_PAWN)

        if white_pieces == 0:
            self.winner = Player.black
            self._is_over_cache = True
            return True
        if black_pieces == 0:
            self.winner = Player.white
            self._is_over_cache = True
            return True

        # Check fortress (expensive - do last)
        if self._is_fortress():
            self.winner = Player.black
            self._is_over_cache = True
            return True

        self._is_over_cache = False
        return False

    def get_legal_moves(self):
        if self._legal_moves_cache is not None:
            return self._legal_moves_cache

        legal_moves = []
        my_pawns = (WHITE_PAWN, KING) if self.next_player == Player.white else (BLACK_PAWN,)
        size = self.board.size
        corners = self.board.corners
        throne = self.board.throne

        # Get piece positions efficiently
        piece_positions = np.where(np.isin(self.board.grid, my_pawns))

        for i in range(len(piece_positions[0])):
            r, c = piece_positions[0][i], piece_positions[1][i]
            piece = self.board.grid[r, c]
            from_pos = Point(r, c)

            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                for distance in range(1, size):
                    nr, nc = r + dr * distance, c + dc * distance
                    if not (0 <= nr < size and 0 <= nc < size):
                        break
                    if self.board.grid[nr, nc] != EMPTY:
                        break

                    to_pos = Point(nr, nc)
                    if piece != KING and (to_pos in corners or to_pos == throne):
                        continue

                    legal_moves.append(Move(from_pos, to_pos))

        self._legal_moves_cache = legal_moves
        return legal_moves

    def find_king(self):
        if self._king_pos_cache is not None:
            return self._king_pos_cache

        king_positions = np.where(self.board.grid == KING)
        if len(king_positions[0]) > 0:
            self._king_pos_cache = Point(king_positions[0][0], king_positions[1][0])
        else:
            self._king_pos_cache = None
        return self._king_pos_cache

    def _check_for_capture(self, move, player):
        opponent_pawn = player.other
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = Point(move.to_pos.row + dr, move.to_pos.col + dc)
            opposite = Point(move.to_pos.row + 2 * dr, move.to_pos.col + 2 * dc)

            if (self.board.is_on_board(neighbor) and
                    self.board.grid[neighbor.row, neighbor.col] == opponent_pawn.value):
                self.will_capture(opposite, player, neighbor)

    def will_capture(self, opposite_point, player, capture_point):
        if self.is_hostile(opposite_point, player):
            self.capture(capture_point)

    def capture(self, point):
        self.board.grid[point.row, point.col] = EMPTY

    def is_hostile(self, point, player):
        if not self.board.is_on_board(point):
            return False

        pawn = self.board.get_pawn_at(point)
        if pawn == player.value:
            return True
        if pawn == KING and player == Player.black:
            return True
        if point in self.board.corners:
            return True
        if (point == self.board.throne and
                (self.board.get_pawn_at(self.board.throne) == EMPTY or
                 self.board.get_pawn_at(self.board.throne) == player.other.value)):
            return True

        return False

    def _is_king_captured(self, king_pos):
        if king_pos is None:
            return True

        attacking_pawns = 0
        for neighbor in king_pos.neighbors():
            if not self.is_hostile(neighbor, Player.black):
                return False
            if self.board.get_pawn_at(neighbor) == BLACK_PAWN:
                attacking_pawns += 1
        return attacking_pawns >= 3

    def _is_fortress(self):
        # Use numpy for faster BFS
        size = self.board.size
        visited = np.zeros((size, size), dtype=bool)
        queue = deque([Point(0, 0), Point(0, size - 1), Point(size - 1, 0), Point(size - 1, size - 1)])

        while queue:
            point = queue.popleft()
            if (not self.board.is_on_board(point) or
                    visited[point.row, point.col]):
                continue

            visited[point.row, point.col] = True
            pawn = self.board.get_pawn_at(point)

            if pawn == WHITE_PAWN or pawn == KING:
                return False
            if pawn == BLACK_PAWN:
                continue

            for neighbor in point.neighbors():
                if (self.board.is_on_board(neighbor) and
                        not visited[neighbor.row, neighbor.col]):
                    queue.append(neighbor)
        return True

if __name__ == '__main__':
    # play a game with random legal moves
    game = GameState.new_game()
    for i in range(1000):
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            break  # No legal moves left, game over
        move = random.choice(legal_moves)
        game = game.apply_move(move)
        #print(game.board)
        print(f"Next player: {game.next_player}, Last move: {game.last_move}, Winner: {game.winner}")

        if game.is_over():
            print(f"Game over! Winner: {game.winner}")
            print(game.board)
            print(i)
            break