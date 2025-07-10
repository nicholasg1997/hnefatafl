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

def decode_action(move):
    from_row = move // (11*4*10)
    rem = move % (11*4*10)

    from_col = rem // (4*10)
    rem = rem % (4*10)

    direction = rem // 10
    distance = (rem % 10) + 1

    return from_row, from_col, direction, distance

def encode_action(from_row, from_col, direction, distance):
    if not (0 <= from_row < 11 and 0 <= from_col < 11):
        raise ValueError("Row and column must be between 0 and 10 inclusive")
    if direction not in [0, 1, 2, 3]:
        raise ValueError("Direction must be one of: 0 (up), 1 (down), 2 (left), 3 (right)")
    if not (0 < distance <= 10):
        raise ValueError("Distance must be between 0 and 10 inclusive")

    return from_row * (11 * 4 * 10) + from_col * (4 * 10) + direction * 10 + (distance - 1)


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