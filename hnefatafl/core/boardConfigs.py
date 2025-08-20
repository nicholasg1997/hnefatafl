from collections import defaultdict

board_07 = [
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 2, 0, 0, 0],
    [1, 1, 2, 3, 2, 1, 1],
    [0, 0, 0, 2, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
]

board_09 = [
    [0, 0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 2, 0, 0, 0, 0],
    [1, 0, 0, 0, 2, 0, 0, 0, 1],
    [1, 1, 2, 2, 3, 2, 2, 1, 1],
    [1, 0, 0, 0, 2, 0, 0, 0, 1],
    [0, 0, 0, 0, 2, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0],
]

board_11 = [
    [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 2, 2, 2, 0, 0, 0, 1],
    [1, 1, 0, 2, 2, 3, 2, 2, 0, 1, 1],
    [1, 0, 0, 0, 2, 2, 2, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
]

fortress_board = [ # for testing fortress detection
    [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
    [1, 1, 1, 0, 0, 2, 0, 0, 1, 1, 1],
    [1, 0, 0, 2, 2, 2, 2, 0, 0, 0, 1],
    [1, 1, 0, 2, 2, 3, 2, 2, 0, 1, 1],
    [1, 0, 0, 0, 2, 2, 2, 0, 0, 0, 1],
    [1, 1, 1, 0, 0, 2, 0, 0, 1, 1, 1],
    [0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
]

def convert_string_to_board(board_string):
    piece_map = {'.': 0, 'B': 1, 'K': 3, 'W': 2}
    board_lines = board_string.strip().split('\n')
    board = []
    for line in board_lines:
        row = [piece_map[char] for char in line.split()]
        board.append(row)
    return board

converted_board = """. B B B B . . B . . .
. B B K B . . . . . .
. . . . . . . . . . B
. . . B . . . . . . .
. . . . . . . . . . B
. . . . . . . B . . .
. . . . . . . . . . .
. . . . . . . . . B .
. . . . . . . . . B B
. B B . . . . B . . B
. . . . B . B . B . .
"""

string_board = convert_string_to_board(converted_board)


BOARD_CONFIGS = defaultdict(dict)



BOARD_CONFIGS[7] = {"board": board_07, "size": 7,
                    "num_attackers": 8, "num_defenders": 5, "name": "Standard 7x7 Board"}

BOARD_CONFIGS[9] = {"board": board_09, "size": 9,
                    "num_attackers": 16, "num_defenders": 9, "name": "Standard 9x9 Board"}

BOARD_CONFIGS[11] = {"board": board_11, "size": 11,
                     "num_attackers": 24, "num_defenders":12 , "name": "Standard 11x11 Board"}

BOARD_CONFIGS[11.1] = {"board": fortress_board, "size": 11,
                     "num_attackers": 24, "num_defenders":12 , "name": "Fortress test Board"}

BOARD_CONFIGS[11.2] = {"board": string_board, "size": 11, 'name': "String Converted Board",}

