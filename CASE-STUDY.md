# What did a tiny GPT learn about Reversi?

Reversi Lab asks whether a small transformer can predict legal moves on a 6×6 board when its only input is the sequence of squares played. The rules engine sees the board and enforces the rules. The model starts with random weights and learns by predicting the next move in simulated games.

## Experimental design

The rules engine generated 30,000 distinct games by picking uniformly among legal moves. It split complete games into 24,000 training, 3,000 validation, and 3,000 test games. A separate 1,000-game stress set preferred moves that flip the most discs 80% of the time. The split excludes duplicate complete histories. Common openings and board positions can still occur in different splits.

The comparison includes a counting baseline that uses the previous token, a GPT restricted to its most recent four input tokens, and a GPT that sees the whole move history. The two GPTs use the same architecture, shuffled training examples, and number of updates. Equal updates and targets can still require different amounts of computation. The validation checkpoint was selected on a fixed subset of 768 games.

The main score is **top-1 legality**, the fraction of positions where the model's most likely token is allowed by the rules. **Legal probability mass** adds the model's raw probabilities for all legal tokens. When several moves are legal, predicting a different legal move from the recorded one still counts. In the browser, the rules-filtered view renormalizes probabilities after removing illegal tokens. All benchmark scores use raw predictions.

## Results

Each cell gives **top-1 legality / mean legal probability mass**. Placement rows exclude forced passes and terminal states. The early, middle, and late buckets start at 0, 10, and 20 disc placements.

| Evaluation slice | Positions | Previous token counts | Last four tokens | Full move history |
| --- | ---: | ---: | ---: | ---: |
| Held-out placements | 95,798 | 41.23% / 24.29% | 58.64% / 38.50% | **82.23% / 65.73%** |
| Early | 30,000 | 59.95% / 35.81% | 87.61% / 68.12% | **92.14% / 76.26%** |
| Middle | 29,989 | 39.44% / 24.99% | 56.97% / 32.85% | **75.81% / 55.02%** |
| Late | 35,809 | 27.04% / 14.04% | 35.76% / 18.43% | **79.30% / 65.88%** |
| Forced passes | 1,674 | 0.00% / 2.68% | 0.06% / 6.85% | **15.59% / 21.68%** |
| Terminal states | 3,000 | 33.47% / 5.03% | 48.47% / 20.80% | **95.80% / 94.99%** |
| Stress-set placements | 31,884 | 44.56% / 24.45% | 59.54% / 38.48% | **81.18% / 64.79%** |

Full history improved top-1 legality by 23.59 percentage points over the last-four model on held-out placements and by 43.54 points late in games. The full-history model kept most of that advantage on the greedy-policy stress set. Forced passes were its weakest category.

Both GPTs have 111,271 parameters and saw 2,572,988 next-token targets over 2,400 updates. Both selected step 2,400 on validation loss. The full-history checkpoint scored 2.092 and the four-token checkpoint scored 2.924. See the [machine-readable benchmark](artifacts/benchmark.json) and the [full](artifacts/full-training.json) and [four-token](artifacts/short-training.json) training logs.

## Two failures you can replay

1. **Late illegal prediction, 30 placements.** Black can only play **B2**. The model gives **F5** 93.66% even though F5 is empty and brackets no White discs. It gives B2 2.33%. With filtering enabled, B2 becomes the only choice and its displayed probability jumps to 100%.
2. **Forced pass, 31 placements.** White has no legal placement, while Black can still move. The model gives the sole empty square, **F6**, 96.91% even though White cannot capture there. It gives **PASS** 2.27%. The filtered view displays PASS at 100%.

## Can we read the board from the model?

A separate experiment froze the GPT and trained a linear classifier to read each square from its final hidden vector. The classifier could choose among empty, mine, and theirs, relative to the player whose turn it was. It trained on positions from 4,000 training games and was scored on 94,472 positions from the 3,000 held-out games.

| Readout input | Individual squares | Mean recall across classes | Entire board correct |
| --- | ---: | ---: | ---: |
| Most common label by square and move count | 61.00% | 53.65% | 0.00% |
| Untrained GPT, same architecture | 64.99% | 60.33% | 3.23% |
| Trained full-history GPT | **78.09%** | **73.21%** | **9.11%** |

The trained model’s internal activity contained more readable board information than either control. The classifier identified empty squares much more reliably than the current player's discs, and it rarely reconstructed the whole board. This test measures what a classifier can read from the model. Determining whether the GPT relies on that information to choose moves would take a separate intervention. [Probe results](artifacts/probe.json) · [Prior Othello-GPT probing work](https://aclanthology.org/2023.blackboxnlp-1.2/)

## What this experiment can establish

A legal move can be strategically poor. The data describes random and partly greedy play, so these results measure rule-following under those two policies. Better scores with longer history would establish that the extra information helped on this benchmark. [Original Othello-GPT study](https://arxiv.org/abs/2210.13382)
