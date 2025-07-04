import unittest
from gameState import GameState, Move, Point, WHITE_PAWN, BLACK_PAWN, KING
from gameTypes import Player

class TestGameState(unittest.TestCase):
    def setUp(self):
        self.game = GameState.new_game(board_size=11)

    def test_initial_state(self):
        self.assertEqual(self.game.next_player, Player.black)
        self.assertIsNone(self.game.winner)

    def test_apply_move_and_switch_player(self):
        move = Move(Point(0, 4), Point(3, 4))
        new_state = self.game.apply_move(move)
        self.assertEqual(new_state.next_player, Player.white)
        self.assertEqual(new_state.board.get_pawn_at(Point(3, 4)), BLACK_PAWN)

    def test_invalid_move_wrong_pawn(self):
        move = Move(Point(0, 0), Point(1, 0))  # Not a black pawn
        with self.assertRaises(ValueError):
            self.game.apply_move(move)

    def test_capture_logic(self):
        # Set up a simple capture scenario
        self.game.board.grid[5, 5] = BLACK_PAWN
        self.game.board.grid[5, 6] = WHITE_PAWN
        self.game.board.grid[0, 0] = KING
        move = Move(Point(5, 7), Point(5, 4))
        self.game.board.grid[5, 7] = WHITE_PAWN
        self.game.next_player = Player.white
        new_state = self.game.apply_move(move)
        self.assertEqual(new_state.board.get_pawn_at(Point(5, 5)), 0)  # Captured

    def test_king_win(self):
        # Move king to a corner
        self.game.board.grid[0, 0] = KING
        self.assertTrue(self.game.is_over())
        self.assertEqual(self.game.winner, Player.white)

if __name__ == '__main__':
    unittest.main()