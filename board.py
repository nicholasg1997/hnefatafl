import numpy as np
from gameTypes import Player, Point
from move import Move

EMPTY = 0
WHITE_PAWN = 2
BLACK_PAWN = 1
KING = 3

class Board:

    def __init__(self, board_size=11):
        assert board_size == 11, "Currently only 11x11 board is supported"
        self.size = board_size
        self.grid = np.zeros((self.size, self.size), dtype=int)
        self.set_up_board()

    def reset(self):
        self.set_up_board()

    def set_up_board(self):
        center = (self.size // 2)
        self.current_player = Player.black
        self.place_king(center)
        self.initialize_black(center)
        self.initialize_white(center)

    def place_king(self, center):
        center_point = Point(center, center)
        self.grid[center_point.row, center_point.col] = KING

    def initialize_black(self, center):
        # initialization of black for 11x11 board
        for i in range(3, self.size - 3):
            self.grid[0, i] = BLACK_PAWN
            self.grid[self.size - 1, i] = BLACK_PAWN
            self.grid[i, 0] = BLACK_PAWN
            self.grid[i, self.size - 1] = BLACK_PAWN
        self.grid[1, center] = BLACK_PAWN
        self.grid[self.size - 2, center] = BLACK_PAWN
        self.grid[center, 1] = BLACK_PAWN
        self.grid[center, self.size - 2] = BLACK_PAWN

    def initialize_white(self, center):
        # initialization of white for 11x11 board
        self.grid[center, center] = KING
        self.grid[center, center - 1] = WHITE_PAWN
        self.grid[center, center + 1] = WHITE_PAWN
        self.grid[center - 1, center] = WHITE_PAWN
        self.grid[center + 1, center] = WHITE_PAWN
        self.grid[center, center - 2] = WHITE_PAWN
        self.grid[center, center + 2] = WHITE_PAWN
        self.grid[center - 2, center] = WHITE_PAWN
        self.grid[center + 2, center] = WHITE_PAWN
        for r_off, c_off in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
             self.grid[center + r_off, center + c_off] = WHITE_PAWN

    def is_on_board(self, point):
        return 0 <= point.row < self.size and 0 <= point.col < self.size

    def get_pawn_at(self, point):
        if not self.is_on_board(point):
            return None
        return self.grid[point.row, point.col]

    def move_pawn(self, move):
        piece = self.get_pawn_at(move.from_pos)
        self.grid[move.from_pos.row, move.from_pos.col] = EMPTY
        self.grid[move.to_pos.row, move.to_pos.col] = piece

    def __str__(self) -> str:
        symbols = {0: '.', 1: 'B', 2: 'W', 3: 'K'}
        rendered = []
        for row in self.grid:
            #print(' '.join(symbols[cell] for cell in row))
            rendered.append(' '.join(symbols[cell] for cell in row))
        return '\n'.join(rendered)





if __name__ == "__main__":
    board = Board()
    board.move_pawn(Move(Point(0, 3), Point(4, 3)))
    print(board)