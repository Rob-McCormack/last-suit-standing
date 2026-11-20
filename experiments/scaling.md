# Training-scale experiment

**Question.** Does a model that gets better at 6-card games become worse at 7-card games?

**The predictions below were written and committed before any runs.** The git history shows the commit time.

## Why ask

Earlier experiments ([size_shift.md](size_shift.md), [size_shift_followup.md](size_shift_followup.md)) suggest models learn the structure of the card count they train on, including which player has the last word, rather than the general rule. If so, more training might entrench that structure, so play at the opposite parity could get *worse* as play at the training size improves. That pattern is called inverse scaling. It matters for safety because it would mean capability on the main task and fragility in a nearby situation grow together, and a test of the main task would not reveal it.

## Design

- Models are trained on 6-card games only, with **250, 500, 1,500 or 4,000 deals**. Smaller sets are prefixes of the larger ones, using the same positions and labels.
- Two model types, each with **3 seeds** (0, 1 and 2): the MLP from `size_shift.py`, and gradient-boosted trees as in `size_shift_followup.py`.
- Testing uses 4 to 8 cards, with the same 200 held-out deals per size as the earlier experiments. The metric is points lost per game against perfect play, both seats averaged.
- **Sanity check:** the MLP at 1,500 deals with seeds 0 and 1 uses exactly the data and settings of `size_shift.py`, so it should reproduce that experiment's 6-card row (2.42 on 6, 11.05 on 7, 8.70 on 8).

## Pre-registered outcomes

All comparisons are between **250 and 4,000 deals**, per model type, using the paired difference per test deal averaged over seeds, with a bootstrap 95% CI.

- **6 cards (training size):** expected to improve. If it does not, the experiment is uninformative for that model type, because more data did not make it more capable.
- **7 cards (opposite parity), the main question.** The outcome is classified as:
  - **Inverse scaling:** the loss increases (CI entirely above 0);
  - **Flat:** the CI includes 0;
  - **Improves:** the loss decreases (CI entirely below 0).
- **8 cards (same parity):** classified the same way.

**My prediction.** For both model types: 6 cards improves; 8 cards improves; **7 cards does not improve** (flat or inverse scaling). The prediction counts as *wrong* for a model type if 7 cards improves with a CI excluding 0.

## Results

Added after the run. Raw output is in `scaling_results.txt`. The sanity check passed exactly: the MLP at 1,500 deals, seeds 0–1, gives 2.42 / 11.05 / 8.70, as in `size_shift.py`.

**Headline.** For the neural network, **more training on 6 cards made play worse at every other card count**, while 6-card play kept improving. The effect is largest and steadiest at 7 cards: the loss rose at every step, from 9.21 to 11.98 points per game, and in all three seeds. Gradient-boosted trees did **not** show this. They improved at the same parity and stayed flat at the opposite parity.

Points lost per game against perfect play, mean of seeds 0–2:

**MLP**

| 6-card training deals | 4 | 5 | *6 (trained)* | 7 | 8 |
| --- | --- | --- | --- | --- | --- |
| 250 | 1.41 | 3.66 | 3.18 | 9.21 | 10.13 |
| 500 | 1.46 | 3.79 | 2.65 | 10.18 | 9.74 |
| 1,500 | 1.40 | 4.28 | 2.42 | 10.97 | 9.48 |
| 4,000 | 1.72 | 4.33 | **2.03** | **11.98** | 11.38 |
| Greedy | 1.29 | 2.90 | 4.41 | 7.26 | 9.88 |

**Gradient-boosted trees**

| 6-card training deals | 4 | 5 | *6 (trained)* | 7 | 8 |
| --- | --- | --- | --- | --- | --- |
| 250 | 1.26 | 3.41 | 2.77 | 8.64 | 8.94 |
| 500 | 1.17 | 3.67 | 2.32 | 8.91 | 8.30 |
| 1,500 | 1.07 | 3.37 | 1.92 | 8.69 | 7.68 |
| 4,000 | 1.07 | 3.36 | **1.78** | 8.95 | 7.75 |
| Greedy | 1.29 | 2.90 | 4.41 | 7.26 | 9.88 |

**Change from 250 to 4,000 deals** (95% CI; negative means it improves), with the pre-registered classification:

| Test size | MLP | Per seed | Trees | Per seed |
| --- | --- | --- | --- | --- |
| 4 | +0.32 [+0.12, +0.51] **inverse scaling** | +0.38, +0.45, +0.13 | −0.19 [−0.30, −0.09] improves | −0.19, −0.22, −0.17 |
| 5 | +0.67 [+0.22, +1.13] **inverse scaling** | +0.88, +0.45, +0.68 | −0.05 [−0.29, +0.20] flat | −0.38, +0.01, +0.23 |
| *6 (trained)* | −1.15 [−1.48, −0.83] improves | −0.73, −1.17, −1.54 | −1.00 [−1.28, −0.71] improves | −0.95, −1.06, −0.97 |
| **7** | **+2.77 [+2.08, +3.54] inverse scaling** | +2.97, +2.25, +3.10 | +0.32 [−0.24, +0.91] flat | +0.03, +0.46, +0.46 |
| 8 | +1.25 [+0.64, +1.87] **inverse scaling** | +1.76, +2.17, −0.20 | −1.19 [−1.81, −0.57] improves | −1.18, −1.38, −1.00 |

### Scoring the predictions

| Prediction | MLP | Trees |
| --- | --- | --- |
| 6 cards improves | confirmed | confirmed |
| 8 cards improves | **wrong**: inverse scaling (2 of 3 seeds) | confirmed |
| 7 cards does not improve | confirmed: **inverse scaling** | confirmed: flat |

For trees, every prediction held. For the MLP, the main prediction held in its stronger form, and the 8-card prediction was wrong: the MLP got worse at 8 cards too.

## What this shows

1. **For the neural network, capability and fragility grew together.** Each increase in training made it better at the game it was trained on and worse at every other card count. A standard check, held-out 6-card deals, would have endorsed the 4,000-deal model as the best one. It is in fact the worst on 4, 5, 7 and 8 cards. At 7 cards it loses 65% more points than an untrained greedy player.
2. **The parity effect remains.** The damage is largest at 7 cards, the nearest size with the opposite parity, and smallest at 4.
3. **The kind of model matters.** Gradient-boosted trees trained on the same data did not degrade. They kept improving at the same parity and stayed flat at the opposite one. Whether more data entrenches a narrow solution depends on the learner, not only on the data.

The broader idea that more training can make models more specialized is familiar in machine learning. What this adds is an exact, pre-registered measurement of it, a clean case where held-out validation on the training task points the wrong way, and a contrast between two model types trained on identical data.

## Limits

Three seeds, 200 test deals per size, two model types with fixed settings, and evaluation against a perfect opponent only. The MLP's 8-card result is the least robust: one seed went the other way. The MLP uses early stopping on a 6-card validation split, so "more training" here means more data at the training size, not more epochs. Inputs use fixed card slots: at 7 and 8 cards a 6-card model reads slots it never saw filled, so part of the damage at those sizes could be an encoding artifact. At 4 and 5 cards no unseen slots are used, and the MLP still got worse. [slot_free.md](slot_free.md) tests the slots. [checkpoint.md](checkpoint.md) tests whether early stopping, rather than the amount of data, causes the 7-card damage.

## Reproduce

```bash
pip install numpy scikit-learn
python3 experiments/scaling.py --sizes 250,500,1500,4000 --seeds 0,1,2 --test-deals 200
```

This takes about 45 minutes on 2 CPU cores. For a quick check, `--sizes 20,40 --seeds 0 --test-deals 10` runs in about 15 seconds.
