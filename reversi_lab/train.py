"""Train a pair of randomly initialized models on the same Reversi targets."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

import torch
from .model import GPT, GPTConfig, IGNORE, game_loss, pack_games, save_model


def choose_device(request: str) -> str:
    if request != "auto":
        return request
    return "mps" if torch.backends.mps.is_available() else "cpu"


@torch.inference_mode()
def validation_loss(model: GPT, games: torch.Tensor, batch_size: int) -> float:
    model.eval()
    total_loss = 0.0
    total_targets = 0
    for batch in games.split(batch_size):
        total_loss += game_loss(model, batch, reduction="sum").item()
        total_targets += (batch[:, 1:] != IGNORE).sum().item()
    model.train()
    return total_loss / total_targets


def training_batch_order(game_count: int, batch_size: int, steps: int, seed: int):
    """Visit all games in shuffled order before starting the next epoch."""
    rng = random.Random(seed)
    indices = list(range(game_count))
    remaining = steps
    while remaining:
        rng.shuffle(indices)
        for offset in range(0, game_count, batch_size):
            yield indices[offset:offset + batch_size]
            remaining -= 1
            if not remaining:
                break


def train(args: argparse.Namespace, dataset: dict, device: str, mode: str) -> dict:
    torch.manual_seed(args.seed)
    config = GPTConfig(history_limit=4 if mode == "short" else None)
    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_games = len(dataset["train"])
    train_games = dataset["train"][:args.train_games]
    validation = dataset["val"][:args.val_games]
    games = pack_games(train_games)
    val_games = pack_games(validation)
    path = args.output / f"{mode}.pt"
    best = float("inf")
    best_step = 0
    validation_history = []
    started = time.monotonic()
    checkpoint_elapsed = 0.0
    training_targets = 0
    training_seconds = 0.0
    timing_started = time.monotonic()
    order = training_batch_order(len(games), args.batch_size, args.steps, args.seed)

    for step, indices in enumerate(order, start=1):
        optimizer.zero_grad(set_to_none=True)
        batch = games[indices]
        loss = game_loss(model, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        training_targets += (batch[:, 1:] != IGNORE).sum().item()

        measure = step == 200 or step == args.steps
        validate = step % args.validate_every == 0 or step == args.steps
        if measure or validate:
            training_value = loss.item()
            if not math.isfinite(training_value):
                raise FloatingPointError(f"{mode} training loss was nonfinite at step {step}")
            training_seconds += time.monotonic() - timing_started

            if step == 200:
                checkpoint_elapsed = training_seconds
                projected = training_seconds * args.steps / step
                print(f"{mode} pilot: {training_seconds:.1f}s for 200 steps, "
                      f"{projected:.1f}s projected for {args.steps} steps", flush=True)
                if projected > 7200 and len(train_games) > total_games // 2:
                    print("Pilot exceeded two hours. Restart both modes with half the games "
                          "and half the steps to preserve epoch coverage.", flush=True)
                    raise TimeoutError("Full-budget training projection exceeds two hours")
            if validate:
                score = validation_loss(model, val_games, args.batch_size)
                if not math.isfinite(score):
                    raise FloatingPointError(f"{mode} validation loss was nonfinite at step {step}")
                print(f"{mode} step {step}/{args.steps} train {training_value:.4f} val {score:.4f}", flush=True)
                validation_history.append({"step": step, "loss": score})
                if score < best:
                    best, best_step = score, step
                    save_model(model, path, {
                        "mode": mode, "step": step, "val_loss": score,
                        "data_sha256": args.data_sha256, "train_games": len(train_games),
                    })
            timing_started = time.monotonic()

    return {
        "mode": mode, "device": device, "seed": args.seed,
        "train_games": len(train_games), "available_train_games": total_games,
        "val_games": len(validation), "steps": args.steps,
        "batch_size": args.batch_size, "learning_rate": args.lr,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "training_targets": training_targets, "best_step": best_step, "best_val_loss": best,
        "pilot_seconds": round(checkpoint_elapsed, 2) if checkpoint_elapsed else None,
        "training_seconds": round(training_seconds, 2),
        "total_seconds": round(time.monotonic() - started, 2),
        "data_sha256": args.data_sha256, "validation": validation_history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("artifacts/data.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--mode", choices=("both", "full", "short"), default="both")
    parser.add_argument("--device", choices=("auto", "mps", "cpu", "cuda"), default="auto")
    parser.add_argument("--steps", type=int, default=2400)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validate-every", type=int, default=300)
    parser.add_argument("--train-games", type=int)
    parser.add_argument("--val-games", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()
    counts = (args.steps, args.batch_size, args.validate_every, args.val_games)
    if (min(counts) < 1 or (args.train_games is not None and args.train_games < 1)
            or not math.isfinite(args.lr) or args.lr <= 0):
        parser.error("Steps, batch size, game counts, validation interval, and learning rate must be positive")

    raw_data = args.data.read_bytes()
    args.data_sha256 = hashlib.sha256(raw_data).hexdigest()
    dataset = json.loads(raw_data)
    device = choose_device(args.device)
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    modes = ("full", "short") if args.mode == "both" else (args.mode,)
    for mode in modes:
        stats = train(args, dataset, device, mode)
        (args.output / f"{mode}-training.json").write_text(json.dumps(stats, indent=2) + "\n")


if __name__ == "__main__":
    main()
