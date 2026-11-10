import unittest

from reversi_lab.game import BLACK, EMPTY, END, PASS, START, WHITE, State


class GameTest(unittest.TestCase):
    def test_opening_and_play(self):
        state = State()
        self.assertEqual(state.legal(), (8, 13, 22, 27))
        self.assertEqual(state.flips(8), (14,))
        next_state = state.play(8)
        self.assertEqual(next_state.board[8], BLACK)
        self.assertEqual(next_state.board[14], BLACK)
        self.assertEqual(next_state.turn, WHITE)
        self.assertEqual(state.board[8], EMPTY)

    def test_move_flips_in_all_eight_directions(self):
        board = [EMPTY] * 36
        targets = []
        for dr, dc in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
            adjacent = (2 + dr) * 6 + 2 + dc
            far = (2 + 2 * dr) * 6 + 2 + 2 * dc
            board[adjacent] = WHITE
            board[far] = BLACK
            targets.append(adjacent)
        state = State(tuple(board))
        self.assertCountEqual(state.flips(14), targets)
        result = state.play(14)
        self.assertTrue(all(result.board[index] == BLACK for index in [14, *targets]))

    def test_gap_and_edge_do_not_bracket(self):
        board = [EMPTY] * 36
        board[1], board[3] = WHITE, BLACK
        board[6], board[7] = WHITE, BLACK
        state = State(tuple(board))
        self.assertNotIn(0, state.legal())
        self.assertNotIn(5, state.legal())

    def test_illegal_moves(self):
        for token in (0, 14, START, PASS, END, -1, 39, True, 8.0):
            with self.subTest(token=token), self.assertRaises(ValueError):
                State().play(token)

    def test_forced_pass_and_game_over_with_empty_squares(self):
        state = State((WHITE, BLACK) + (EMPTY,) * 34)
        self.assertEqual(state.legal(), (PASS,))
        other = state.play(PASS)
        self.assertEqual(other.board, state.board)
        self.assertEqual(other.turn, WHITE)
        self.assertEqual(other.legal(), (2,))
        terminal = other.play(2)
        self.assertEqual(terminal.legal(), (END,))
        with self.assertRaises(ValueError):
            terminal.play(PASS)
        final = terminal.play(END)
        self.assertTrue(final.ended)
        self.assertEqual(final.legal(), ())
        with self.assertRaises(ValueError):
            final.play(END)


if __name__ == "__main__":
    unittest.main()
