from hnefatafl.utils.utils import Point, decode_action, encode_action, calculate_end_position

class Move:
    def __init__(self, from_pos, to_pos):
        self.from_pos = from_pos
        self.to_pos = to_pos

    @classmethod
    def from_encoded(cls, encoded_move, board_size=11):
        from_row, from_col, direction, distance = decode_action(encoded_move, board_size)
        from_pos = Point(from_row, from_col)
        to_pos = calculate_end_position(from_col, from_row, direction, distance)
        return cls(from_pos, to_pos)

    def encode(self, board_size=11):
        if self.from_pos.row == self.to_pos.row:  # Horizontal move
            distance = abs(self.from_pos.col - self.to_pos.col)
            direction = 2 if self.from_pos.col < self.to_pos.col else 3

        else:
            distance = abs(self.from_pos.row - self.to_pos.row)
            direction = 0 if self.from_pos.row > self.to_pos.row else 1

        return encode_action(self.from_pos.row, self.from_pos.col, direction, distance, board_size=board_size)

    def __eq__(self, other):
        return isinstance(other, Move) and self.from_pos == other.from_pos and self.to_pos == other.to_pos

    def __hash__(self):
        return hash((self.from_pos, self.to_pos))

    def __str__(self):
        return f"Move(from={self.from_pos}, to={self.to_pos})"

if __name__ == "__main__":
    # Example usage
    move = Move(Point(5, 10), Point(5, 9))
    encoded_move = move.encode()
    print(f"Encoded Move: {encoded_move}")

    decoded_move = Move.from_encoded(2630)
    print(f"Decoded Move: {decoded_move}")
