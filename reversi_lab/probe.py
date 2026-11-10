"""Read relative board state from a frozen GPT with a linear classifier."""

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .game import END, SQUARES, START, State
from .model import GPT, IGNORE, load_model, pack_games

CLASSES = ("empty", "mine", "theirs")
MAX_PLACEMENTS = SQUARES - 4


@dataclass
class BoardLabels:
    select: torch.Tensor
    boards: torch.Tensor
    placements: torch.Tensor


def board_labels(games: Sequence[Sequence[int]]) -> BoardLabels:
    """Label valid prefixes after the first placement and before game over."""
    select, boards, placement_counts = [], [], []
    for game in games:
        state = State()
        placements = 0
        for following in game[1:]:
            keep = placements > 0 and following != END
            select.append(keep)
            if keep:
                board = [0 if not disc else 1 if disc == state.turn else 2 for disc in state.board]
                boards.append(board)
                placement_counts.append(placements)
            state = state.play(following)
            placements += following < START
    return BoardLabels(
        torch.tensor(select, dtype=torch.bool),
        torch.tensor(boards, dtype=torch.long).reshape(-1, SQUARES),
        torch.tensor(placement_counts, dtype=torch.long),
    )


@torch.no_grad()
def game_activations(model: GPT, games: Sequence[Sequence[int]]) -> torch.Tensor:
    """Capture the final normalized hidden vector, before vocabulary projection."""
    packed = pack_games(games)
    device = next(model.parameters()).device
    inputs = packed[:, :-1].masked_fill(packed[:, :-1] == IGNORE, START).to(device)
    activations = []
    handle = model.output[0].register_forward_hook(
        lambda _module, _inputs, output: activations.append(output)
    )
    try:
        model(inputs)
    finally:
        handle.remove()
    return activations[0][(packed[:, 1:] != IGNORE).to(device)].cpu()


def features(model: GPT, games: Sequence[Sequence[int]], select: torch.Tensor) -> torch.Tensor:
    batches = [game_activations(model, games[offset:offset + 64])
               for offset in range(0, len(games), 64)]
    return torch.cat(batches)[select]


def train_probe(activations: torch.Tensor, boards: torch.Tensor, seed: int,
                epochs: int) -> nn.Linear:
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)
    # The output has three independent scores for each board square.
    probe = nn.Linear(activations.size(1), SQUARES * len(CLASSES))
    optimizer = torch.optim.AdamW(probe.parameters(), lr=0.003)
    for _ in range(epochs):
        for indices in torch.randperm(len(activations), generator=rng).split(1024):
            logits = probe(activations[indices]).reshape(-1, len(CLASSES))
            loss = F.cross_entropy(logits, boards[indices].flatten())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return probe.eval()


@torch.no_grad()
def probe_predictions(probe: nn.Linear, activations: torch.Tensor) -> torch.Tensor:
    predictions = [probe(batch).reshape(-1, SQUARES, len(CLASSES)).argmax(dim=-1)
                   for batch in activations.split(2048)]
    return torch.cat(predictions)


def majority_by_placement(train: BoardLabels, test: BoardLabels) -> torch.Tensor:
    """Predict the most frequent class for a square at this placement count."""
    labels = F.one_hot(train.boards, num_classes=len(CLASSES))
    counts = torch.zeros(MAX_PLACEMENTS + 1, SQUARES, len(CLASSES), dtype=torch.long)
    counts.index_add_(0, train.placements, labels)
    global_majority = labels.sum(dim=0).argmax(dim=-1)
    prior = torch.where(counts.sum(dim=-1) > 0, counts.argmax(dim=-1), global_majority)
    return prior[test.placements]


def score(predictions: torch.Tensor, labels: BoardLabels) -> dict:
    correct = predictions == labels.boards
    recalls = {name: correct[labels.boards == index].float().mean().item()
               for index, name in enumerate(CLASSES)}
    stages = {
        "early": labels.placements < 10,
        "middle": (labels.placements >= 10) & (labels.placements < 20),
        "late": labels.placements >= 20,
    }
    return {
        "square_accuracy": correct.float().mean().item(),
        "balanced_accuracy": sum(recalls.values()) / len(recalls),
        "recall": recalls,
        "exact_board_accuracy": correct.all(dim=1).float().mean().item(),
        "square_accuracy_by_stage": {
            stage: correct[mask].float().mean().item() if mask.any() else None
            for stage, mask in stages.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("artifacts/data.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/full.pt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/probe.json"))
    parser.add_argument("--train-games", type=int, default=4000)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.train_games, args.epochs) < 1:
        parser.error("Train game count and epoch count must be positive")

    torch.set_num_threads(4)
    raw_data = args.data.read_bytes()
    data_sha = hashlib.sha256(raw_data).hexdigest()
    data = json.loads(raw_data)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["data_sha256"] != data_sha:
        parser.error("Checkpoint training data differs from the probe dataset")
    full_model = load_model(args.checkpoint)
    if full_model.config.history_limit is not None:
        parser.error("The probe expects a full-history model")
    # Use a different seed from the main run for the untrained control.
    torch.manual_seed(args.seed + 1)
    untrained_model = GPT(full_model.config).eval()
    train_games = data["train"][:min(args.train_games, checkpoint["train_games"])]
    test_games = data["test"]
    train = board_labels(train_games)
    test = board_labels(test_games)

    report = {
        "data_sha256": data_sha,
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "seed": args.seed,
        "untrained_model_seed": args.seed + 1,
        "epochs": args.epochs,
        "train_games": len(train_games),
        "test_games": len(test_games),
        "train_positions": len(train.boards),
        "test_positions": len(test.boards),
        "classes": CLASSES,
        "results": {"square_and_placement_majority": score(majority_by_placement(train, test), test)},
    }
    for name, model in (("untrained_gpt", untrained_model), ("trained_gpt", full_model)):
        print(f"Fitting linear readouts on {name}", flush=True)
        train_activations = features(model, train_games, train.select)
        probe = train_probe(train_activations, train.boards, args.seed, args.epochs)
        test_activations = features(model, test_games, test.select)
        report["results"][name] = score(probe_predictions(probe, test_activations), test)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
