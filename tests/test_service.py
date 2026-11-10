"""Prediction frames validate game histories and expose raw model output."""
import unittest
from unittest.mock import patch

from reversi_lab.game import END, START, VOCAB, State
from reversi_lab.serve import prediction_frame


class PredictionFrameTests(unittest.TestCase):
    @patch("reversi_lab.model.predict")
    def test_opening_preserves_raw_distribution(self, predict):
        predict.return_value = [1 / VOCAB] * VOCAB
        model = object()
        frame = prediction_frame([START], model)
        predict.assert_called_once_with(model, [START])
        self.assertEqual(frame["legal"], State().legal())
        self.assertEqual(frame["probabilities"], predict.return_value)

    def test_invalid_history_is_rejected(self):
        for history in (None, [], [False], [START, False], [START, START], [START, 0]):
            with self.subTest(history=history), self.assertRaises(ValueError):
                prediction_frame(history, None)

    @patch("reversi_lab.model.predict")
    def test_ended_game_has_no_prediction(self, predict):
        state = State()
        history = [START]
        while not state.ended:
            token = state.legal()[0]
            state = state.play(token)
            history.append(token)
        self.assertEqual(history[-1], END)
        frame = prediction_frame(history, None)
        self.assertIsNone(frame["probabilities"])
        self.assertEqual(frame["legal"], ())
        predict.assert_not_called()
