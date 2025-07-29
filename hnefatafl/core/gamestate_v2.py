from collections import deque
import numpy as np
import random
from numba import njit
from collections import Counter

from hnefatafl.core.gameTypes import Player, Point
from hnefatafl.core.board import Board
from hnefatafl.core.move import Move

EMPTY = 0
WHITE_PAWN = 2
BLACK_PAWN = 1
KING = 3


@njit
def _get_legal_moves(grid, my_pawns, board_size, corners, throne):
    legal_moves = []

    for r in range(board_size):
        for c in range(board_size):
            piece = grid[r, c]
            if piece in my_pawns:
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    for distance in range(1, board_size):
                        to_r = r + dr * distance
                        to_c = c + dc * distance

                        if not (0 <= to_r < board_size and 0 <= to_c < board_size):
                            break
                        if grid[to_r, to_c] != EMPTY:
                            break

                        is_restricted = False
                        if piece != KING:
                            if to_r == throne[0] and to_c == throne[1]:
                                is_restricted = True
                            for cr, cc in corners:
                                if to_r == cr and to_c == cc:
                                    is_restricted = True
                                    break
                        if is_restricted:
                            continue
                        legal_moves.append((r, c, to_r, to_c))
    return legal_moves


class GameState:
    def __init__(self, board, next_player, previous, move, history=None, pawn_history=None):
        self.board = board
        self.next_player = next_player
        self.previous = previous
        self.last_move = move
        self.winner = None

        turn_detection = 3 # number of turns back to detect repetition
        self.pawn_history = pawn_history if pawn_history is not None else deque(maxlen=2*turn_detection)

        self.history = history if history is not None else Counter()
        current_hash = (self.board.grid.tobytes(), self.next_player)
        self.history[current_hash] += 1
        self.repetition_hit = False

    @classmethod
    def new_game(cls, board_size=11):
        return cls(Board(board_size), Player.black, previous=None, move=None, history=Counter())

    def apply_move(self, move):
        new_board = Board(self.board.size)
        new_board.grid = np.copy(self.board.grid)
        # ensure that the pawn being moved is the correct player's pawn
        moving_pawn = self.board.get_pawn_at(move.from_pos)
        if self.next_player == Player.white:
            if moving_pawn not in [WHITE_PAWN, KING]:
                print(f"Invalid move: {move} for player {self.next_player}")
                print(self.board)
                raise ValueError("Invalid move: not a white pawn or king")
        else:
            if moving_pawn != BLACK_PAWN:
                print(f"Invalid move: {move} for player {self.next_player}")
                print(self.board)
                raise ValueError("Invalid move: not a black pawn")

        new_board.move_pawn(move)

        next_state = GameState(new_board, self.next_player.other, self,
                               move, self.history.copy(), self.pawn_history.copy())

        next_state._update_pawn_history()
        next_state._check_for_capture(move, self.next_player)
        next_state.is_over()

        return next_state

    def is_over(self):
        if self.winner is not None:
            return True

        if self.previous is not None and len(self.pawn_history) > 0:
            current_pawn_hash = self._get_pawn_positions(self.next_player.other)
            for pawn_hash, player in self.pawn_history:
                if player == self.next_player.other and pawn_hash == current_pawn_hash:
                    #print(f"move repetition detected, game over for player {self.next_player.other}")
                    self.winner = self.next_player
                    self.repetition_hit = True
                    return True

        current_hash = (self.board.grid.tobytes(), self.next_player)
        if self.history[current_hash] >= 3:
            #print(f"repetition detected, game over for player {self.next_player.other}")
            self.winner = self.next_player
            self.repetition_hit = True
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

        if not self.get_legal_moves():
            self.winner = self.next_player.other
            return True

        return False

    def _get_pawn_positions(self, player):
        grid = self.board.grid
        empty_grid = np.zeros_like(grid)
        player_pawns = [WHITE_PAWN, KING] if player == Player.white else [BLACK_PAWN,]
        for row in range(grid.shape[0]):
            for col in range(grid.shape[1]):
                if grid[row, col] in player_pawns:
                    empty_grid[row, col] = 1

        #print(f"Calculating pawn positions for player {player}, grid:\n{empty_grid}")
        return hash(empty_grid.tobytes())

    def _update_pawn_history(self):
        """
        Update the pawn history with the current state of the board for the next player.
        This is used to detect repetitions in pawn positions.
        """
        if self.previous is not None:
            current_pawn_hash = self._get_pawn_positions(self.next_player)
            self.pawn_history.append((current_pawn_hash, self.next_player))

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
        # TODO: add Ko style repetition detection

        corners = np.array([[p.row, p.col] for p in self.board.corners])
        throne = np.array([self.board.throne.row, self.board.throne.col])

        my_pawns = (WHITE_PAWN, KING) if self.next_player == Player.white else (BLACK_PAWN,)

        moves  = _get_legal_moves(self.board.grid,
                                  my_pawns,
                                  self.board.size,
                                  corners,
                                  throne)

        if not moves:
            print("No legal moves found.")
            print(self.board)


        return [Move(Point(fr, fc), Point(tr, tc)) for fr, fc, tr, tc in moves]


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