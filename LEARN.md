# Build your own understanding

Set aside one session each week for code you type and can explain, and another for AI-assisted work. Treat the completed project as a reference implementation. Keep a short note on what you predicted, what you observed, and what you would change.

## Weeks 1–2 · Predict a square

Open [MicroGPT in Colab](https://colab.research.google.com/drive/1vyN5zo6rqUp_dYNbT4Yrco66zuWCZKoN?usp=sharing). Run it once, find the line that sets the number of training steps, and change that number. Compare the loss and samples. Then put the code aside.

Number a tic-tac-toe board from 0 to 8, left to right and top to bottom. X plays first. Use these four partial games.

```python
games = [
    [0, 4, 1, 8, 2],
    [2, 4, 1, 8, 0],
    [6, 4, 0, 2, 3],
    [0, 4, 8, 2, 6],
]
```

1. On paper, count which moves immediately follow square 4. Your counts should be square 1 twice, and squares 0 and 8 once each.
2. Write a Python loop to get the same result. Generalize it to count successors of every square. Convert the counts to probabilities and print them.
3. Write a function that replays a sequence and prints the board, using move parity to choose X or O. Try histories `[0, 4]` and `[2, 4]`. Your counting predictor sees the same final move, yet the set of empty squares differs.
4. Try predicting the next move on a new history. Ask which candidate the counts prefer and whether that candidate is legal.

## Weeks 3–4 · Become the referee

Trace the 6×6 Reversi opening on paper. Pick a legal move, explain which disc flips and why, then write an assertion for it before reading the rules tests. Next, find a forced pass in the demo and explain why it is a pass instead of game over. Inspect one saved game and draw the boundary between the input tokens and the next-token answer.

## Weeks 5–6 · Follow one training step

Use the small run in the README. Pick one game prefix and trace it through the tokenizer, embeddings, attention, logits, and loss. Before a second run, predict what would happen if training were stopped after one update. Explain how evaluation differs from training and why the rules engine never goes into the model input.

## Weeks 7–8 · Read the errors

Choose two held-out failures. Replay each board by hand or with the rules engine. Write one sentence explaining exactly why the move was illegal. Compare the full-history and four-token rows in the benchmark. Ask what a difference in legal probability mass tells you even when top-1 legality is unchanged.

## Weeks 9–10 · Give the demo

Practice a two-minute walkthrough. Show one successful prediction, one illegal suggestion, and what the rules filter changes. Then explain one thing you wrote, one thing AI helped with, and one result that surprised you. Use the case study as a starting point and add your own experiment or reflection before presenting the work as a portfolio project.
