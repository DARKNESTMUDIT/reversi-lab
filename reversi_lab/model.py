"""A GPT that learns next-move probabilities from Reversi token sequences."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .game import START, VOCAB

IGNORE = -100


@dataclass(frozen=True)
class GPTConfig:
    width: int = 64
    heads: int = 4
    layers: int = 2
    context: int = 96
    history_limit: int | None = None


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        width = config.width
        self.heads = config.heads
        self.attention_norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.project = nn.Linear(width, width)
        self.feedforward = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, 4 * width), nn.GELU(),
            nn.Linear(4 * width, width),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        qkv = self.qkv(self.attention_norm(x))
        q, k, v = qkv.reshape(batch, length, 3, self.heads, width // self.heads).unbind(2)
        q, k, v = (value.transpose(1, 2) for value in (q, k, v))
        # Causal attention lets each position see itself and earlier tokens.
        attention = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.project(attention.transpose(1, 2).reshape(batch, length, width))
        return x + self.feedforward(x)


class GPT(nn.Module):
    def __init__(self, config: GPTConfig = GPTConfig()):
        super().__init__()
        self.config = config
        self.tokens = nn.Embedding(VOCAB, config.width)
        self.positions = nn.Embedding(config.context, config.width)
        self.blocks = nn.Sequential(*(Block(config) for _ in range(config.layers)))
        self.output = nn.Sequential(nn.LayerNorm(config.width), nn.Linear(config.width, VOCAB))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        length = tokens.size(1)
        limit = self.config.history_limit or self.config.context
        if not 1 <= length <= limit:
            raise ValueError(f"Expected a token sequence of length 1 to {limit}")
        positions = torch.arange(length, device=tokens.device)
        return self.output(self.blocks(self.tokens(tokens) + self.positions(positions)))


def pack_games(games: Sequence[Sequence[int]]) -> torch.Tensor:
    """Right-pad games with an ignored label. START is used only for input padding."""
    length = max(map(len, games))
    return torch.tensor(
        [list(game) + [IGNORE] * (length - len(game)) for game in games], dtype=torch.long
    )


def suffix_batch(games: torch.Tensor, limit: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Make each target's last few tokens an independent transformer input."""
    starts = games.new_full((games.size(0), limit - 1), IGNORE)
    windows = torch.cat((starts, games[:, :-1]), dim=1).unfold(1, limit, 1)
    targets = games[:, 1:]
    valid = targets != IGNORE
    windows = windows[valid]
    # For early moves, shift real tokens left so each window starts at position zero.
    lengths = (windows != IGNORE).sum(dim=1)
    positions = torch.arange(limit, device=windows.device)
    source = (positions[None, :] + limit - lengths[:, None]).clamp(max=limit - 1)
    inputs = windows.gather(1, source).masked_fill(positions[None, :] >= lengths[:, None], START)
    return inputs, lengths - 1, targets[valid]


def game_logits(model: GPT, games: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one logit vector per actual next-move target, in game order."""
    device = next(model.parameters()).device
    targets = games[:, 1:]
    limit = model.config.history_limit
    if limit is None:
        inputs = games[:, :-1].masked_fill(games[:, :-1] == IGNORE, START).to(device)
        return model(inputs)[(targets != IGNORE).to(device)], targets[targets != IGNORE].to(device)

    inputs, last, targets = suffix_batch(games, limit)
    logits = model(inputs.to(device))
    rows = torch.arange(len(last), device=device)
    return logits[rows, last.to(device)], targets.to(device)


def game_loss(model: GPT, games: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """Compute next-token loss while ignoring right padding."""
    if model.config.history_limit is not None:
        logits, targets = game_logits(model, games)
        return F.cross_entropy(logits, targets, reduction=reduction)
    device = next(model.parameters()).device
    inputs = games[:, :-1].masked_fill(games[:, :-1] == IGNORE, START).to(device)
    targets = games[:, 1:].to(device)
    logits = model(inputs)
    return F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=IGNORE, reduction=reduction)


def save_model(model: GPT, path: Path, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save({"config": asdict(model.config), "state_dict": model.state_dict(), **metadata}, temporary)
    temporary.replace(path)


def load_model(path: str | Path, device: str = "cpu") -> GPT:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = GPT(GPTConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()


@torch.inference_mode()
def predict(model: GPT, history: Sequence[int]) -> list[float]:
    limit = model.config.history_limit or model.config.context
    device = next(model.parameters()).device
    tokens = torch.tensor([history[-limit:]], dtype=torch.long, device=device)
    return model(tokens)[0, -1].softmax(dim=-1).tolist()
