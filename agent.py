import random
from gameTypes import Point
from move import Move

class Agent():
    def select_move(selfself, game_state):
        """
        Selects a move based on the current game state.
        This method should be overridden by subclasses to implement specific strategies.

        :param game_state: The current state of the game.
        :return: A Move object representing the selected move.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def diagnostic(self, game_state):
        """
        Provides diagnostic information about the agent's decision-making process.
        This method can be overridden by subclasses to provide specific diagnostics.

        :param game_state: The current state of the game.
        :return: A string containing diagnostic information.
        """
        return {}


class RandomAgent(Agent):
    def select_move(self, game_state):
        """
        Selects a random legal move from the current game state.

        :param game_state: The current state of the game.
        :return: A Move object representing the selected move.
        """
        legal_moves = game_state.get_legal_moves()
        if not legal_moves:
            print(legal_moves)
            raise ValueError("No legal moves available.")
        return random.choice(legal_moves)


def point_from_coord(coord):
    """
    Converts a coordinate string (e.g., 'A1') to a Point object.

    :param coord: A string representing the coordinate.
    :return: A Point object corresponding to the coordinate.
    """
    col = ord(coord[0].upper()) - ord('A')
    row = int(coord[1]) - 1
    return Point(col, row)


class HumanAgent(Agent):
    def select_move(self, game_state):
        """
        Prompts the user to input a move based on the current game state.

        :param game_state: The current state of the game.
        :return: A Move object representing the user's selected move.
        """
        print("Current board:")
        print(game_state.board)

        move_input = input("Enter your move (e.g., 'A1 B2'): ")
        from_pos, to_pos = move_input.split()
        move = Move(point_from_coord(from_pos), point_from_coord(to_pos))

        legal_moves = game_state.get_legal_moves()
        if move not in legal_moves:
            raise ValueError("Invalid move. Please try again.")
        return move


