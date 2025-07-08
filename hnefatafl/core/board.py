import numpy as np
from hnefatafl.core.gameTypes import Player, Point
from hnefatafl.core.move import Move

from hnefatafl.core.boardConfigs import BOARD_CONFIGS

EMPTY = 0
WHITE_PAWN = 2
BLACK_PAWN = 1
KING = 3

class Board:

    def __init__(self, board=11):
        self.board_config = BOARD_CONFIGS[board]
        self.size = self.board_config['size']
        self.grid = np.zeros((self.size, self.size), dtype=int)
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
            rendered.append(' '.join(symbols[cell] for cell in row))
        return '\n'.join(rendered)




if __name__ == "__main__":
    board = Board()
    print(board)