"""Generate deterministic, disjoint splits of full Reversi game histories."""

import argparse
import json
from pathlib import Path
import random

from .game import END, START, State


def play_game(rng: random.Random, greedy_probability: float = 0.0) -> tuple[int, ...]:
    history = [START]
    state = State()
    while not state.ended:
        moves = state.legal()
        if len(moves) == 1:
            token = moves[0]
        elif greedy_probability and rng.random() < greedy_probability:
            flip_counts = {move: len(state.flips(move)) for move in moves}
            maximum = max(flip_counts.values())
            token = rng.choice([move for move, count in flip_counts.items() if count == maximum])
        else:
            token = rng.choice(moves)
        history.append(token)
        state = state.play(token)
    return tuple(history)


def generate(games: int, stress: int, seed: int) -> dict[str, list[tuple[int, ...]]]:
    if games < 10 or stress < 0:
        raise ValueError("Generate at least 10 main games and zero or more stress games.")
    rng = random.Random(seed)
    seen: set[tuple[int, ...]] = set()

    def unique_games(count: int, greedy_probability: float) -> list[tuple[int, ...]]:
        result = []
        while len(result) < count:
            history = play_game(rng, greedy_probability)
            if history not in seen:
                seen.add(history)
                result.append(history)
        return result

    main = unique_games(games, 0.0)
    train_end, val_end = games * 8 // 10, games * 9 // 10
    return {
        "train": main[:train_end],
        "val": main[train_end:val_end],
        "test": main[val_end:],
        "stress": unique_games(stress, 0.8),
    }


def validate(splits: dict[str, list[list[int]]]) -> None:
    """Raise ValueError for duplicate, incomplete, or illegal game histories."""
    seen: set[tuple[int, ...]] = set()
    for split in ("train", "val", "test", "stress"):
        for index, history in enumerate(splits[split]):
            context = f"{split} game {index + 1}"
            key = tuple(history)
            if key in seen:
                raise ValueError(f"{context}: duplicate history")
            if len(history) < 2 or history[0] != START or history[-1] != END:
                raise ValueError(f"{context}: expected START and END boundaries")
            state = State()
            for token in history[1:]:
                try:
                    state = state.play(token)
                except ValueError as error:
                    raise ValueError(f"{context}: {error}") from error
            seen.add(key)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/data.json"))
    parser.add_argument("--games", type=int, default=30_000)
    parser.add_argument("--stress", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validate", action="store_true", help="Check the existing --output file.")
    args = parser.parse_args()
    if args.validate:
        splits = json.loads(args.output.read_text())
        validate(splits)
    else:
        splits = generate(args.games, args.stress, args.seed)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(splits, separators=(",", ":")) + "\n")
    print(", ".join(f"{name}={len(games)}" for name, games in splits.items()), flush=True)


if __name__ == "__main__":
    main()
