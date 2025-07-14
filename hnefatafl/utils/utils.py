from hnefatafl.core.gameTypes import Player, Point

PAWN_TO_CHAR = {
    None: '.',
    Player.black: 'B',
    Player.white: 'W',
}

def print_move(board, point):
    char = PAWN_TO_CHAR.get(board[point.row, point.col], '.')
    print(f"Move to {point} with pawn {char}")

def print_board(board):
    for row in board:
        print(' '.join(PAWN_TO_CHAR.get(cell, '.') for cell in row))
    print()

def point_from_coords(row, col):
    return Point(row, col)

def coords_from_point(point):
    return point.row, point.col

def decode_action(move, board_size):
    max_dist = board_size - 1
    moves_per_square = 4 * max_dist
    moves_per_row = board_size * moves_per_square

    from_row = move // moves_per_row
    rem = move % moves_per_row

    from_col = rem // moves_per_square
    rem = rem % moves_per_square

    direction = rem // max_dist
    distance = (rem % max_dist) + 1

    return from_row, from_col, direction, distance

def encode_action(from_row, from_col, direction, distance, board_size):
    max_dist = board_size - 1
    if not (0 <= from_row < board_size and 0 <= from_col < board_size):
        raise ValueError(f"Row and column must be between 0 and {board_size - 1} inclusive")
    if direction not in [0, 1, 2, 3]:
        raise ValueError("Direction must be one of: 0 (up), 1 (down), 2 (left), 3 (right)")
    if not (0 < distance <= max_dist):
        raise ValueError(f"Distance must be between 1 and {max_dist} inclusive")

    moves_per_square = 4 * max_dist
    moves_per_row = board_size * moves_per_square

    return from_row * moves_per_row + from_col * moves_per_square + direction * max_dist + (distance - 1)


def calculate_end_position(from_col, from_row, direction, distance):
    if direction == 0:  # Up
        return Point(from_row - distance, from_col)
    elif direction == 1:  # Down
        return Point(from_row + distance, from_col)
    elif direction == 2:  # Left
        return Point(from_row, from_col + distance)
    elif direction == 3:  # Right
        return Point(from_row, from_col - distance)
    else:
        raise ValueError("Invalid direction")