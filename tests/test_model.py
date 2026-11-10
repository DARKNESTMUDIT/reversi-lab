"""Check training and live prediction agree for both context lengths."""

import unittest

import torch

from reversi_lab.evaluate import CountingModel, Score
from reversi_lab.game import END, START, VOCAB
from reversi_lab.model import GPT, GPTConfig, game_logits, game_loss, pack_games, predict


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.games = [[START, 8, 9, END], [START, 13, 12, 11, 10, 17, END]]

    def test_causal_full_model_is_consistent_with_prefix_prediction(self):
        model = GPT(GPTConfig()).eval()
        batch = pack_games(self.games)
        logits, targets = game_logits(model, batch)
        self.assertEqual(targets.tolist(), [8, 9, END, 13, 12, 11, 10, 17, END])
        expected = [predict(model, game[:index]) for game in self.games
                    for index in range(1, len(game))]
        torch.testing.assert_close(logits.softmax(dim=-1), torch.tensor(expected))
        torch.testing.assert_close(game_loss(model, batch),
                                   torch.nn.functional.cross_entropy(logits, targets))

    def test_short_model_cannot_read_earlier_tokens(self):
        model = GPT(GPTConfig(history_limit=4)).eval()
        self.assertEqual(predict(model, [START, 22, 12, 11, 10, 17]),
                         predict(model, [START, 13, 12, 11, 10, 17]))
        batch = pack_games(self.games)
        logits, _ = game_logits(model, batch)
        expected = [predict(model, game[:index]) for game in self.games
                    for index in range(1, len(game))]
        torch.testing.assert_close(logits.softmax(dim=-1), torch.tensor(expected))

    def test_raw_metrics_count_probability_on_legal_moves(self):
        probabilities = [0.1, 0.2, 0.5, 0.2]
        score = Score()
        score.add(probabilities, [0, 1])
        score.add(probabilities, [2, 3])
        self.assertEqual(score.report()["top1_legal"], 0.5)
        self.assertAlmostEqual(score.report()["legal_mass"], 0.5)

    def test_counting_model_smooths_unseen_transitions(self):
        model = CountingModel(self.games)
        self.assertAlmostEqual(sum(model.table[START]), 1)
        self.assertEqual(len(model.table[START]), VOCAB)
        self.assertGreater(model.table[START][8], model.table[START][0])


if __name__ == "__main__":
    unittest.main()
