import unittest

from reversi_lab.data import generate, validate
from reversi_lab.game import END, START


class DataTest(unittest.TestCase):
    def setUp(self):
        self.splits = generate(20, 3, seed=42)

    def test_deterministic_disjoint_splits_replay(self):
        self.assertEqual(self.splits, generate(20, 3, seed=42))
        self.assertEqual(
            {name: len(games) for name, games in self.splits.items()},
            {"train": 16, "val": 2, "test": 2, "stress": 3},
        )
        validate(self.splits)

    def test_duplicate_is_rejected(self):
        self.splits["val"][0] = self.splits["train"][0]
        with self.assertRaisesRegex(ValueError, "duplicate history"):
            validate(self.splits)

    def test_truncated_and_illegal_histories_are_rejected(self):
        original = self.splits["train"][0]
        self.splits["train"][0] = original[:-1]
        with self.assertRaisesRegex(ValueError, "expected START and END"):
            validate(self.splits)
        self.splits["train"][0] = (START, END)
        with self.assertRaisesRegex(ValueError, "Illegal move"):
            validate(self.splits)


if __name__ == "__main__":
    unittest.main()
