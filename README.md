# Reversi Lab

**Can a tiny GPT learn which moves are legal from move histories alone?**

Reversi Lab pairs a 6×6 Reversi rules engine with a decoder-only transformer trained from random weights. The transformer receives only move tokens. The rules engine generates games and grades the predictions. The browser shows both the model's raw choices and the choices that remain after applying the rules.

## Explore

[Open the public demo](https://darknestmudit.github.io/reversi-lab/) or open `site/index.html` locally. Step through the bundled predictions, including a successful move, an illegal suggestion, and a forced pass. The replays run offline. For new positions, start the local server with the included checkpoint and choose **Live play**.

```sh
uv venv --python 3.12
uv pip install -r requirements.txt
.venv/bin/python -m reversi_lab.serve --checkpoint artifacts/full.pt
```

Open <http://127.0.0.1:8000>. Play a legal move or inspect the model’s top five choices. In filtered mode the visible probabilities are renormalized over legal moves. On a forced pass, for example, the filtered result is 100% even if the model assigned `PASS` very little probability. The server binds to loopback and needs no API key.

## Reproduce

Run these commands from the repository root after setting up the environment above.

```sh
# Rules, data, model, and prediction service tests
.venv/bin/python -m unittest discover -s tests -v

# Small CPU training check with a separate output directory
.venv/bin/python -m reversi_lab.data --games 100 --stress 10 --output artifacts/smoke-data.json
.venv/bin/python -m reversi_lab.train --data artifacts/smoke-data.json --output artifacts/smoke --steps 8 --batch-size 8 --device cpu

# Full dataset and split validation
.venv/bin/python -m reversi_lab.data --games 30000 --stress 1000 --seed 42
.venv/bin/python -m reversi_lab.data --validate

# Reproduce the benchmark using the included checkpoints
.venv/bin/python -m reversi_lab.evaluate --device cpu

# Reproduce the board-state readout experiment
.venv/bin/python -m reversi_lab.probe

# Retrain both released architectures from fresh weights
.venv/bin/python -m reversi_lab.train
```

Training uses the Apple GPU when PyTorch supports it and otherwise uses the CPU. Retraining can produce slightly different numbers across hardware and PyTorch versions. The bundled checkpoint and fixed dataset are the inputs for reproducing the released benchmark.

## Read the project

- [Case study](CASE-STUDY.md) explains the experiment, results, and limitations.
- [Learning guide](LEARN.md) turns the project into ten weeks of exercises, starting with basic Python and a counting predictor.
- [Karpathy’s MicroGPT](https://karpathy.github.io/2026/02/12/microgpt/) is the starting point for understanding training from scratch.
- [Li et al.’s Othello-GPT](https://github.com/likenneth/othello_world) inspired the move-history experiment.
