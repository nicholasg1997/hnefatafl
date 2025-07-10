from collections import deque
import numpy as np
import random
import copy
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
            self.winner = Player.black
            return True

        flattened_board = self.board.grid.flatten()
        counts = Counter(flattened_board)
        if counts[WHITE_PAWN] == 0 and counts[KING] == 0:
            self.winner = Player.black
            return True
        if counts[BLACK_PAWN] == 0:
            self.winner = Player.white
            return True

        return False

    def get_legal_moves(self):
        """
        Calculate all legal moves for the current player based on the state of the board. A move
        is considered legal if it adheres to the rules of movement, avoiding obstacles and respecting
        any specific pawn or king constraints.

        The function inspects all pawns of the current player and determines their potential destinations
        on the board by iterating through possible directions (up, down, left, right). It handles special
        cases such as restricted movement for non-king pawns near the throne or corners.

        :return: A list of all unique legal moves available for the current player
        :rtype: list[Move]
        """
        legal_moves = []
        my_pawns = (WHITE_PAWN, KING) if self.next_player == Player.white else (BLACK_PAWN,)
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

        if not legal_moves:
            print("No legal moves found.")
            print(self.board)

        return list(legal_moves)

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
        #print(f"Next player: {game.next_player}, Last move: {game.last_move}, Winner: {game.winner}")

        if game.is_over():
            print(f"Game over! Winner: {game.winner}")
            print(game.board)
            print(i)
            break