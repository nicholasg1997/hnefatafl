import numpy as np
from gameTypes import Player, Point
from utils import decode, encode

EMPTY = 0
WHITE_PAWN = 2
BLACK_PAWN = 1
KING = 3

class Board:

    def __init__(self, board_size=11):
        assert board_size == 11, "Currently only 11x11 board is supported"
        self.size = board_size

    def reset(self):
        self.set_up_board()
        self.print_board()

    def set_up_board(self):
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.current_player = Player.black
        self.place_king()
        self.initialize_black()
        self.initialize_white()

    def place_king(self):
        center = (self.size // 2)
        center_point = Point(center, center)
        self.board[center_point.row, center_point.col] = KING

    def initialize_black(self):
        # TODO: make the black pawn logic work with different sized boards
        center = (self.size // 2)
        for i in range(3, self.size - 3):
            self.board[0, i] = BLACK_PAWN
            self.board[self.size - 1, i] = BLACK_PAWN
            self.board[i, 0] = BLACK_PAWN
            self.board[i, self.size - 1] = BLACK_PAWN
        self.board[1, center] = BLACK_PAWN
        self.board[self.size - 2, center] = BLACK_PAWN
        self.board[center, 1] = BLACK_PAWN
        self.board[center, self.size - 2] = BLACK_PAWN

    def initialize_white(self):
        center = (self.size // 2)
        center_point = Point(center, center)

        pawns_to_place = (self.size + 1)
        points = [center_point]

        while pawns_to_place > 0:
            neighbours = []
            for x in points:
                # Get all neighbors of the current point
                neighbours.extend(x.neighbors())

            for n in neighbours:
                if self.is_on_board(n) and self.board[n.row, n.col] == 0:
                    self.board[n.row, n.col] = WHITE_PAWN  # Place white pawn
                    pawns_to_place -= 1
                    points.append(n)
                    if pawns_to_place == 0:
                        break

    def is_on_board(self, point):
        return 0 <= point.row < self.size and 0 <= point.col < self.size

    def get_pawn_at(self, point):
        if not self.is_on_board(point):
            return None
        return self.board[point.row, point.col]

    def move_pawn(self, player, move):
        # will need to check: pawn of current player, valid move, capture logic
        # move is legal if it doesn't go off the board and doesn't hop over another pawn
        from_row, from_col, direction, distance = decode(move)
        pawn = self.get_pawn_at(Point(from_row, from_col))
        if pawn != (WHITE_PAWN if player == Player.white else BLACK_PAWN):
            raise ValueError("Invalid pawn for the current player")
        if direction == 0:  # Up
            new_row = from_row - distance
            new_col = from_col
        elif direction == 1:  # Down
            new_row = from_row + distance
            new_col = from_col
        elif direction == 2:  # Left
            new_row = from_row
            new_col = from_col - distance
        elif direction == 3:  # Right
            new_row = from_row
            new_col = from_col + distance
        else:
            raise ValueError("Invalid direction")

        new_point = Point(new_row, new_col)
        if not self.is_on_board(new_point):
            raise ValueError("Move goes off the board")
        if self.get_pawn_at(new_point) != EMPTY:
            raise ValueError("Cannot move to a point that is already occupied by another pawn")
        self.board[from_row, from_col] = EMPTY
        self.board[new_point.row, new_point.col] = pawn
        self._check_for_capture(new_point, player)

    def will_capture(self, neighbor_point, player, capture_point):
        new_point = neighbor_point
        if self.board[new_point.row, new_point.col] == player.value:
            self.capture(capture_point)
        elif not self.is_on_board(new_point):
            self.capture(capture_point)

    def capture(self, point):
        self.board[point.row, point.col] = EMPTY


    def _check_for_capture(self, new_point, player):
        opponent_pawn = player.other.value
        # Check if the new point is adjacent to an opponent's pawn
        d = 0 # 0:up, 1:down, 2:left, 3:right
        for neighbor in new_point.neighbors():
            if (self.is_on_board(neighbor) and
                    self.board[neighbor.row, neighbor.col] == opponent_pawn):

                if d == 0:  # Up
                    self.will_capture(Point(neighbor.row - 1, neighbor.col), player, neighbor)

                elif d == 1:  # Down
                    self.will_capture(Point(neighbor.row + 1, neighbor.col), player, neighbor)

                elif d == 2:  # Left
                    self.will_capture(Point(neighbor.row, neighbor.col - 1), player, neighbor)

                elif d == 3:  # Right
                    self.will_capture(Point(neighbor.row, neighbor.col + 1), player, neighbor)

            d += 1


    def get_valid_moves(self, player):
        pass

    def is_king(self, point):
        return self.board[point.row, point.col] == KING

    def print_board(self):
        symbols = {0: '.', 1: 'B', 2: 'W', 3: 'K'}
        for row in self.board:
            print(' '.join(symbols[cell] for cell in row))
        print()




if __name__ == "__main__":
    board = Board(11)
    board.reset()
    board.move_pawn(Player.black, encode(3, 0, 3, 4))
    board.print_board()
    board.move_pawn(Player.black, encode(0, 6, 1, 3))
    board.print_board()