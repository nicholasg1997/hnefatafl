from collections import deque
import numpy as np

from gameTypes import Player, Point
from board import Board
from move import Move

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

    @classmethod
    def new_game(cls, board_size=11):
        return cls(Board(board_size), Player.white, None, None)

    def apply_move(self, move):
        new_board = Board(self.board.size)
        new_board.grid = np.copy(self.board.grid)
        new_board.move_pawn(move)

        next_state = GameState(new_board, self.next_player.other, self, move)
        next_state._check_for_capture(move.to_pos, self.next_player, move)
        next_state.is_over()
        return next_state

    def is_over(self):
        if self.winner is not None:
            return True

        king_pos = self.find_king()
        if self._is_king_captured(king_pos):
            self.winner = Player.black
            return True

        if king_pos is None: # redundant check just in case
            self.winner = Player.black
            return True

        size = self.board.size
        corners = [Point(0, 0), Point(0, size - 1), Point(size - 1, 0), Point(size - 1, size - 1)]

        if king_pos in corners:
            self.winner = Player.white
            return True

        if self._is_fortress():
            print("Fortress detected")
            self.winner = Player.black
            return True

        return False

    def get_legal_moves(self):
        legal_moves = []
        my_pawns = (WHITE_PAWN, KING) if self.next_player == Player.white else (BLACK_PAWN, KING)
        size = self.board.size
        corners = self.board.corners
        throne = self.board.throne

        for r in range(size):
            for c in range(size):
                piece = self.board.grid[r, c]
                if piece in my_pawns:
                    from_pos = Point(r, c)
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        for distance in range(1, size):
                            to_pos = Point(r + dr * distance, c + dc * distance)
                            if not self.board.is_on_board(to_pos):
                                break
                            if self.board.get_pawn_at(to_pos) != EMPTY:
                                break
                            if piece != KING and (to_pos in corners or to_pos == throne):
                                continue


                            move = Move(from_pos, to_pos)
                            legal_moves.append(move)


    def will_capture(self, neighbor_point, player, capture_point):
        new_point = neighbor_point
        if self.is_hostile(new_point, player):
            self.capture(capture_point)
            print(f"Capture at {capture_point} by {player}.")


    def _check_for_capture(self, point, player, move):
        opponent_pawn = player.other.value
        # Check if the new point is adjacent to an opponent's pawn
        d = 0 # 0:up, 1:down, 2:left, 3:right
        for neighbor in point.neighbors():
            if (self.board.is_on_board(neighbor) and
                    self.board.grid[neighbor.row, neighbor.col] == opponent_pawn):

                if d == 0:  # Up
                    self.will_capture(Point(neighbor.row - 1, neighbor.col), player, neighbor)
                elif d == 1:  # Down
                    self.will_capture(Point(neighbor.row + 1, neighbor.col), player, neighbor)
                elif d == 2:  # Left
                    self.will_capture(Point(neighbor.row, neighbor.col - 1), player, neighbor)
                elif d == 3:  # Right
                    self.will_capture(Point(neighbor.row, neighbor.col + 1), player, neighbor)

            d += 1

    def capture(self, point):
        self.board.grid[point.row-1, point.col-1] = EMPTY

    def is_hostile(self, point, player):
        # Check if the point is occupied by an opponent's pawn or is a corner point or center point
        if not self.board.is_on_board(point):
            return False
        pawn = self.board.get_pawn_at(point)
        if pawn == player.other.value:
            return True
        if pawn == KING and player == Player.black:
            # King is only hostile to black pawns
            return True

        throne = self.board.throne
        corners = self.board.corners
        if point in corners:
            return True

        if point == throne and (self.board.get_pawn_at(throne) == EMPTY or self.board.get_pawn_at(throne) == player.other.value):
            return True

        if pawn == EMPTY:
            return False

        return False

    def _is_king_captured(self, king_pos):
        # Check if the king is captured by checking if it is surrounded by hostile pawns
        attacking_pawns = 0
        for neighbor in king_pos.neighbors():
            if not self.is_hostile(neighbor, self.next_player):
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
        pass

    def _is_exit_fort(self):
        pass

if __name__ == '__main__':
    game_state = GameState.new_game()
    print(game_state.board)
    print(game_state.board.grid)
    print("\n")
    new_state = game_state.apply_move(Move(Point(0, 4), Point(3, 4)))
    print(new_state.board)
    print("\n")
    new_state = new_state.apply_move(Move(Point(0, 6), Point(3, 6)))
    print(new_state.board)







