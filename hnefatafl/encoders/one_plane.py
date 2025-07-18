import numpy as np

from hnefatafl.encoders.base import Encoder
from hnefatafl.core.gameTypes import Point, Player
from hnefatafl.core.move import Move

EMPTY = 0
BLACK_PAWN = 1
WHITE_PAWN = 2
KING = 3

class OnePlaneEncoder(Encoder):
    def __init__(self, board_size=11):
        self.board_width, self.board_height = board_size, board_size
        self.num_planes = 1
        self.move_space_size = self.board_width * self.board_height * 4 * (self.board_width - 1)  # 4 directions, each with (width-1) possible moves

    def name(self):
        return 'one_plane'

    def encode(self, game_state):
        board_matrix = np.zeros((self.board_width, self.board_height))
        next_player = game_state.next_player
        player_pieces = [KING, WHITE_PAWN] if next_player == Player.white else [BLACK_PAWN]
        for r in range(self.board_height):
            for c in range(self.board_width):
                p = Point(r,c)
                piece = game_state.board.get_pawn_at(p)
                if piece == EMPTY:
                    continue
                if piece in player_pieces:
                    board_matrix[r, c] = 1
                else:
                    board_matrix[r, c] = -1
        return board_matrix[np.newaxis, :, :]

    def encode_point(self, point):
        return self.board_width * point.row + point.col

    def encode_move(self, move):
        return move.encode()

    def decode_point_index(self, index):
        row = index // self.board_width
        col = index % self.board_width
        return Point(row, col)

    def decode_move_index(self, move_index):
        return Move.from_encoded(move_index, self.board_width)

    def num_points(self):
        return self.board_width * self.board_height

    def num_moves(self):
        return self.move_space_size

    def shape(self):
        return self.num_planes, self.board_height, self.board_width



if __name__ == "__main__":
    from hnefatafl.core.gameState import GameState
    from hnefatafl.core.gameTypes import Player
    game_state = GameState.new_game()
    encoder = OnePlaneEncoder()
    encoded_state = encoder.encode(game_state)
    print(game_state.board)
    print("Encoded Game State:")
    print(encoded_state)



