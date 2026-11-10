"""The probe labels each hidden vector with its current board, before the next move."""

import random
import unittest

import torch

from reversi_lab.data import play_game
from reversi_lab.game import State
from reversi_lab.model import GPT
from reversi_lab.probe import board_labels, features, game_activations


class ProbeTests(unittest.TestCase):
    def test_board_labels_and_activations_use_the_same_prefix(self):
        torch.set_num_threads(1)
        history = play_game(random.Random(42))
        labels = board_labels([history])
        state = State().play(history[1])
        expected = [0 if not disc else 1 if disc == state.turn else 2 for disc in state.board]
        self.assertEqual(labels.boards[0].tolist(), expected)
        self.assertEqual(labels.placements[0].item(), 1)
        self.assertFalse(labels.select[0].item())
        self.assertFalse(labels.select[-1].item())

        model = GPT().eval()
        activations = features(model, [history], labels.select)
        self.assertEqual(len(activations), len(labels.boards))
        # The first selected activation can be computed with only START and
        # the first move as inputs, so it cannot see the next move or board label.
        torch.testing.assert_close(activations[0], game_activations(model, [history[:3]])[-1])


if __name__ == "__main__":
    unittest.main()
