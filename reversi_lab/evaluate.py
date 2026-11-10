"""Benchmark raw next-move predictions and export recorded demo replays."""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Sequence

import torch

from .game import END, PASS, START, VOCAB, State
from .model import GPT, game_logits, load_model, pack_games
from .train import choose_device

MODELS = ("counting", "full", "short")
BUCKETS = ("all_moves", "early", "middle", "late", "forced_pass", "terminal")


def square(token: int) -> str:
    if token < START:
        return f"{chr(ord('a') + token % 6)}{token // 6 + 1}"
    return {START: "start", PASS: "pass", END: "end"}[token]


def top_choice(probabilities: Sequence[float]) -> int:
    return max(range(len(probabilities)), key=probabilities.__getitem__)


class CountingModel:
    """A smoothed table of next-token counts for each previous token."""

    def __init__(self, games: Sequence[Sequence[int]]):
        counts = defaultdict(Counter)
        for game in games:
            for previous, following in zip(game, game[1:]):
                counts[previous][following] += 1
        self.table = []
        for previous in range(VOCAB):
            total = sum(counts[previous].values()) + VOCAB
            self.table.append([(counts[previous][token] + 1) / total for token in range(VOCAB)])


@dataclass
class Score:
    positions: int = 0
    legal_top1: int = 0
    legal_mass: float = 0.0

    def add(self, probabilities: Sequence[float], legal: Sequence[int]) -> None:
        self.positions += 1
        self.legal_top1 += top_choice(probabilities) in legal
        self.legal_mass += sum(probabilities[move] for move in legal)

    def report(self) -> dict:
        return {
            "positions": self.positions,
            "top1_legal": self.legal_top1 / self.positions if self.positions else None,
            "legal_mass": self.legal_mass / self.positions if self.positions else None,
        }


@dataclass
class Example:
    game: Sequence[int]
    focus: int
    legal: Sequence[int]
    choice: int
    recorded_next: int
    confidence: float
    legal_mass: float
    placements: int

    def report(self) -> dict:
        return {
            "history": [square(token) for token in self.game[:self.focus + 1]],
            "placements": self.placements,
            "raw_top": square(self.choice),
            "raw_top_probability": self.confidence,
            "legal_moves": [square(token) for token in self.legal],
            "legal_mass": self.legal_mass,
            "recorded_next": square(self.recorded_next),
        }


@dataclass
class Examples:
    picks: dict[str, Example] = field(default_factory=dict)

    def consider(self, game: Sequence[int], focus: int, legal: Sequence[int],
                 probabilities: Sequence[float], placements: int) -> None:
        choice = top_choice(probabilities)
        confidence = probabilities[choice]
        if legal == (PASS,):
            key = "forced-pass"
        elif legal == (END,):
            return
        elif placements >= 20 and choice not in legal:
            key = "late-failure"
        elif placements <= 9 and choice in legal:
            key = "early-success"
        else:
            return

        existing = self.picks.get(key)
        example = Example(game, focus, legal, choice, game[focus + 1], confidence,
                          sum(probabilities[token] for token in legal), placements)
        # Prefer a forced-pass failure for the second annotated failure.
        rank = self.rank(key, example)
        if existing is None or rank > self.rank(key, existing):
            self.picks[key] = example

    @staticmethod
    def rank(key: str, example: Example) -> tuple[bool, float]:
        if key == "early-success":
            return example.placements >= 3, example.confidence
        return example.choice not in example.legal, example.confidence


@torch.inference_mode()
def probabilities_for_games(model: GPT, games: Sequence[Sequence[int]]) -> list[list[float]]:
    logits, _ = game_logits(model, pack_games(games))
    return logits.softmax(dim=-1).tolist()


def stage(placements: int) -> str:
    if placements < 10:
        return "early"
    if placements < 20:
        return "middle"
    return "late"


def evaluate(games: Sequence[Sequence[int]], counting: CountingModel,
             full: GPT, short: GPT, examples: Examples | None = None) -> dict:
    scores = {name: {bucket: Score() for bucket in BUCKETS} for name in MODELS}
    batch_size = 64
    for offset in range(0, len(games), batch_size):
        batch = games[offset:offset + batch_size]
        full_probabilities = iter(probabilities_for_games(full, batch))
        short_probabilities = iter(probabilities_for_games(short, batch))

        for game in batch:
            state = State()
            placements = 0
            for focus, (previous, following) in enumerate(zip(game, game[1:])):
                legal = state.legal()
                if legal == (PASS,):
                    buckets = ("forced_pass",)
                elif legal == (END,):
                    buckets = ("terminal",)
                else:
                    buckets = ("all_moves", stage(placements))

                predictions = {
                    "counting": counting.table[previous],
                    "full": next(full_probabilities),
                    "short": next(short_probabilities),
                }
                for name, probabilities in predictions.items():
                    for bucket in buckets:
                        scores[name][bucket].add(probabilities, legal)

                if examples is not None:
                    examples.consider(game, focus, legal, predictions["full"], placements)
                state = state.play(following)
                placements += following < START

    return {name: {bucket: scores[name][bucket].report() for bucket in BUCKETS} for name in MODELS}


SCENARIOS = {
    "early-success": ("An early legal prediction", "The model favors a legal move early in an unseen game."),
    "late-failure": ("A late illegal prediction", "Late in the game, the model's favorite token is illegal. Compare the raw ranking with the filtered ranking."),
    "forced-pass": ("A forced pass", "The current player has no square to play. Compare the model's probabilities with the engine's only legal action."),
}


def recorded_replay(model: GPT, key: str, example: Example) -> dict:
    title, description = SCENARIOS[key]
    frames = []
    state = State()
    for index, probabilities in enumerate(probabilities_for_games(model, [example.game])):
        following = example.game[index + 1]
        frames.append({
            "history": example.game[:index + 1], "board": state.board, "turn": state.turn,
            "legal": state.legal(), "probabilities": probabilities, "next": following,
        })
        state = state.play(following)
    frames.append({
        "history": example.game, "board": state.board, "turn": state.turn,
        "legal": state.legal(), "probabilities": None, "next": None,
    })
    return {"id": key, "title": title, "description": description, "focus": example.focus, "frames": frames}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("artifacts/data.json"))
    parser.add_argument("--models", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/benchmark.json"))
    parser.add_argument("--replays", type=Path, default=Path("site/replays.js"))
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()
    raw_data = args.data.read_bytes()
    dataset = json.loads(raw_data)
    data_sha256 = hashlib.sha256(raw_data).hexdigest()
    torch.set_num_threads(4)
    device = choose_device(args.device)
    full_path = args.models / "full.pt"
    short_path = args.models / "short.pt"
    checkpoint_metadata = [torch.load(path, map_location="cpu", weights_only=True)
                           for path in (full_path, short_path)]
    if any(item["data_sha256"] != data_sha256 for item in checkpoint_metadata):
        parser.error("Checkpoint training data differs from the evaluation dataset")
    if checkpoint_metadata[0]["train_games"] != checkpoint_metadata[1]["train_games"]:
        parser.error("Checkpoints were trained on different numbers of games")
    counting = CountingModel(dataset["train"][:checkpoint_metadata[0]["train_games"]])
    full, short = load_model(full_path, device), load_model(short_path, device)
    if full.config.history_limit is not None or short.config.history_limit != 4:
        parser.error("Expected a full-history checkpoint and a four-token checkpoint")

    examples = Examples()
    results = {"data_sha256": data_sha256, "train_games": checkpoint_metadata[0]["train_games"]}
    for split in ("test", "stress"):
        print(f"Scoring {split}: {len(dataset[split]):,} games", flush=True)
        results[split] = evaluate(dataset[split], counting, full, short, examples if split == "test" else None)
    results["examples"] = {key: value.report() for key, value in examples.picks.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")

    replays = [recorded_replay(full, key, examples.picks[key]) for key in SCENARIOS if key in examples.picks]
    args.replays.parent.mkdir(parents=True, exist_ok=True)
    args.replays.write_text("window.REVERSI_REPLAYS = " + json.dumps(replays, separators=(",", ":")) + ";\n")
    print(f"Wrote {args.output} and {len(replays)} replays to {args.replays}", flush=True)


if __name__ == "__main__":
    main()
