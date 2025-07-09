import numpy as np

from hnefatafl.encoders.base import Encoder
from hnefatafl.core.gameTypes import Point, Player

EMPTY = 0
BLACK_PAWN = 1
WHITE_PAWN = 2
KING = 3

class OnePlaneEncoder(Encoder):
    def __init__(self, board_size=11):
        self.board_width, self.board_height = board_size, board_size
        self.num_planes = 1

    def name(self):
        return 'one_plane'

    def encode(self, game_state):
        board_matrix = np.zeros((self.board_width, self.board_height))
        next_player = game_state.next_player
        print(f"Next player: {next_player}")
        player_pieces = [KING, WHITE_PAWN] if next_player == Player.white else [BLACK_PAWN]
        print(player_pieces)
        for r in range(self.board_height):
            for c in range(self.board_width):
                p = Point(r,c)
                piece = game_state.board.get_pawn_at(p)
                print(f"r: {r}, c: {c}, piece: {piece}")
                if piece == EMPTY:
                    continue
                if piece in player_pieces:
                    board_matrix[r, c] = 1
                else:
                    board_matrix[r, c] = -1
        return board_matrix

    def encode_points(self, point):
        pass

    def encode_point_index(self, index):
        pass

    def num_points(self):
        pass

    def shape(self):
        pass



if __name__ == "__main__":
    from hnefatafl.core.gameState import GameState
    from hnefatafl.core.gameTypes import Player
    game_state = GameState.new_game()
    encoder = OnePlaneEncoder()
    encoded_state = encoder.encode(game_state)
    print(game_state.board)
    print("Encoded Game State:")
    print(encoded_state)



