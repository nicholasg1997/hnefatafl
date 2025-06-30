import enum
from collections import namedtuple

class Player(enum.Enum):
    black = 1
    white = 2

    @property
    def other(self):
        return Player.black if self == Player.white else Player.white


class Point(namedtuple("Point", "row col")):
    def neighbors(self):
        return [
            Point(self.row - 1, self.col),  # Up
            Point(self.row + 1, self.col),  # Down
            Point(self.row, self.col - 1),  # Left
            Point(self.row, self.col + 1),  # Right
        ]

    def vert_neighbors(self):
        return [
            Point(self.row - 1, self.col),  # Up
            Point(self.row + 1, self.col),  # Down
        ]

    def hor_neighbors(self):
        return [
            Point(self.row, self.col - 1),  # Left
            Point(self.row, self.col + 1),  # Right
        ]

    def __deepcopy__(self, memodict={}):
        return self
