# Slot-free encoding control

**Question.** Are findings 1 and 2 caused by how positions are fed to the models? Every earlier experiment uses fixed card slots: 8 row slots and 4 stack slots per player, zero when empty. A 6-card game never fills some of those slots, so at 7 or 8 cards a model reads inputs it never trained on. A review by Gemini suggested this may explain the results. Here the same experiments are rerun with inputs that have no fixed card slots.

**The predictions below were written and committed before any runs.** The git history shows the commit time.

## The new encoding

Everything else is unchanged from `size_shift.py`, `size_shift_followup.py` and `scaling.py`: the same solver, training deals, rollouts, suit relabelling, models, one-step policy, test deals (200 per size) and metric (points lost per game against perfect play, both seats).

The inputs:

- **Row, no positions:** count and rank sum per suit; count per rank; over all neighboring pairs, the sums of the lower rank, the higher rank, same-suit pairs and same-rank pairs; the left and right end cards (rank and suit). The end cards always exist, so they are never padding.
- **Stacks, as P1 minus P2 differences:** rank sum per suit, count per rank, stack size, and whether the player holds a pair.
- **Turn:** cards left in the row and who offers next.
- Left out: "offers still to come" (a duplicate of cards left), and "who makes the last offer" (that is the positive control in `size_shift_followup.md`, not part of the base encoding).

Cards left is kept on purpose. Without it a model cannot work out who makes the last offer, so the test could not tell "failed to learn the rule" from "had no way to learn it". A model that treats 8 cards like 7 because 7 is the nearest size it knows is still showing the failure that finding 2 describes; this test removes only the slot artifact.

**Fairness check.** The new encoding must play 6-card games about as well as the old one: 6-card loss within 0.5 points of the old encoding for the same model and training size. If it is not, the encoding lost too much information and the run is reported as uninformative for that model.

## Design

- **Part A (finding 2).** 6-only versus mix 6+7, 1,500 deals per size, MLP and gradient-boosted trees, seeds 0, 1, 2. Tested on 4–8 cards.
- **Part B (finding 1).** The training-scale experiment: 250, 500, 1,500 and 4,000 six-card deals (prefixes), MLP and trees, seeds 0, 1, 2. Tested on 4–8 cards.
- Comparisons use the paired difference per test deal, averaged over seeds, with a bootstrap 95% CI over deals. Per-seed values are reported too. "Yes" means the CI is entirely above 0.

## Pre-registered predictions

| | Prediction | Gemini | Mine |
| --- | --- | --- | --- |
| a | Part A, MLP: mix 6+7 loses more than 6-only on 8 cards | Yes | Yes |
| b | Part A, trees: the same | Yes | Yes |
| c | Part B, MLP, 250 → 4,000 deals: 7-card loss rises | Yes | Yes |
| d | Part B, MLP: 8-card loss rises | No | No |
| e4 | Part B, MLP: 4-card loss rises | No | No |
| e5 | Part B, MLP: 5-card loss rises | Unsure | Yes |

Gemini's reasoning, in short: the slots drive the 8-card scaling and the 4-card change; the 7-card effects come from the parity difference and survive. Gemini expects (a) and (c) to be smaller than before. It is least sure about the MLP on (a) and (d), and calls (e5) close to a coin flip.

**How the findings are judged.**

- **Finding 2** (adding 7-card data hurts 8-card play) stands if (a) and (b) are both Yes. If either is No, the finding is rewritten as a slot artifact for that model type.
- **Finding 1** (more training, more fragility) stands at 7 cards if (c) is Yes. If (d) is No, the README's claim of damage at every other card count is narrowed to the sizes where it survives. If (c) is No, finding 1 is withdrawn as a slot artifact.

## Results

Added after the run. Raw output is in `slot_free_results.txt`. The run took 3,319 seconds. Six-card position counts match `scaling_results.txt` exactly (seed 0: 252,726; seed 1: 253,005; seed 2: 252,987), so the deals and rollouts are the same and only the inputs changed. This machine had no earlier scikit-learn; the run used scikit-learn 1.9.1.

**Fairness check.** Trees pass. Their 6-card loss is 2.63 / 2.32 / 1.90 / 1.80, against the old 2.77 / 2.32 / 1.92 / 1.78. The MLP fails. At 500 deals it loses 3.45, against the old 2.65, which is 0.80 points worse than the 0.5 limit. The other three MLP sizes are inside the limit: 3.48 vs 3.18, 2.78 vs 2.42, and 2.18 vs 2.03. Those three are the sizes the scored comparisons use. The rule still makes the whole MLP run uninformative, because one training size missed. Scores for (a), (c), (d), (e4) and (e5) are reported below and do not decide the findings.

**Part A.** Points lost per game, mean of seeds 0–2. The 6-only row is Part B's 1,500-deal model.

| | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- |
| MLP, 6-only | 1.25 | 6.05 | 2.78 | 11.47 | 6.42 |
| MLP, mix 6+7 | 1.10 | 2.06 | 2.71 | 4.58 | 13.54 |
| Trees, 6-only | 0.84 | 5.56 | 1.90 | 10.62 | 6.02 |
| Trees, mix 6+7 | 1.15 | 1.58 | 2.48 | 3.03 | 8.88 |
| Greedy | 1.29 | 2.90 | 4.41 | 7.26 | 9.88 |

Mix minus 6-only on 8 cards (positive means the mix loses more):

| | 8 cards | Per seed |
| --- | --- | --- |
| MLP (a) | +7.12 [+6.07, +8.16] | +7.13, +6.45, +7.78 |
| Trees (b) | +2.87 [+2.16, +3.63] | +3.72, +2.79, +2.10 |

**Part B.** Points lost per game, mean of seeds 0–2.

| MLP, 6-card deals | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- |
| 250 | 1.41 | 5.72 | 3.48 | 10.63 | 6.98 |
| 500 | 1.46 | 5.72 | 3.45 | 11.74 | 7.87 |
| 1,500 | 1.25 | 6.05 | 2.78 | 11.47 | 6.42 |
| 4,000 | 0.85 | 6.11 | 2.18 | 11.62 | 5.46 |
| Greedy | 1.29 | 2.90 | 4.41 | 7.26 | 9.88 |

| Trees, 6-card deals | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- |
| 250 | 0.98 | 5.14 | 2.63 | 9.95 | 7.32 |
| 500 | 1.03 | 5.42 | 2.32 | 10.33 | 6.68 |
| 1,500 | 0.84 | 5.56 | 1.90 | 10.62 | 6.02 |
| 4,000 | 0.77 | 5.70 | 1.80 | 10.15 | 5.14 |
| Greedy | 1.29 | 2.90 | 4.41 | 7.26 | 9.88 |

Change from 250 to 4,000 deals. Negative means play improved.

| Test size | MLP | Per seed | Trees | Per seed |
| --- | --- | --- | --- | --- |
| 4 | −0.55 [−0.73, −0.39] improves | −0.52, −0.64, −0.50 | −0.21 [−0.35, −0.10] improves | −0.14, −0.34, −0.17 |
| 5 | +0.39 [−0.21, +0.97] flat | +0.27, +0.08, +0.82 | +0.55 [+0.06, +1.05] inverse scaling | +0.27, +0.09, +1.30 |
| 6 | −1.30 [−1.61, −1.00] improves | −1.25, −1.48, −1.18 | −0.82 [−1.06, −0.59] improves | −0.77, −0.75, −0.95 |
| 7 | +1.00 [+0.40, +1.61] inverse scaling | +0.89, +0.70, +1.40 | +0.21 [−0.46, +0.90] flat | +0.20, +0.32, +0.11 |
| 8 | −1.51 [−2.03, −1.00] improves | −1.07, −2.09, −1.38 | −2.18 [−2.70, −1.69] improves | −2.37, −1.96, −2.20 |

The MLP's 7-card loss does not rise at every step: 10.63, 11.74, 11.47, 11.62. The pre-registered comparison is the endpoints only.

### Scored predictions

"Yes" means the 95% CI is entirely above 0.

| | Observed | Gemini | Mine |
| --- | --- | --- | --- |
| a | Yes | Yes | Yes |
| b | Yes | Yes | Yes |
| c | Yes | Yes | Yes |
| d | No | No | No |
| e4 | No | No | No |
| e5 | No | Unsure | Yes |

(e5) is the miss on Mine's list. The 5-card loss rose by 0.39 points, and in every seed, but the interval includes 0, so the score is No. Gemini left (e5) as Unsure. Gemini and Mine match on (a) through (e4).

### What this means under the decision rules

**Finding 2** stands only when (a) and (b) are both Yes. (b) is Yes, on the model that passed the fairness check. For the trees, adding 7-card games still makes 8-card play worse, by 2.87 points, in all three seeds. The earlier tree result was +5.35, so the effect is smaller, as Gemini expected, and it is still there. A 6-only tree also still loses more on 7 cards (10.62) than on 8 (6.02). (a) scored Yes, by 7.12 points in all three seeds, and the Part A model itself is inside the fairness band (2.78 vs 2.42 at 1,500 deals). The MLP fails the check at 500 deals, so (a) does not count. Finding 2 does not stand for both model types. The rewrite, "a slot artifact for that model type," applies when the answer is No. (b) is Yes, and (a) is uninformative, so neither model type is rewritten that way.

**Finding 1** stands at 7 cards only when (c) is Yes. The interval is above 0: +1.00 points, in all three seeds, against +2.77 with the old encoding. The MLP run is uninformative, so (c) does not confirm the finding. Withdrawing it as a slot artifact requires (c) to be No. (d) is No: 8-card loss fell by 1.51 points, in all three seeds, where the old encoding had risen by 1.25. (e4) is No: 4-card loss fell by 0.55. The rule would then narrow the README line "worse at every other card count" to the sizes that still get worse. Those two scores are MLP scores, and the fairness check blocks that edit.

No README change follows from this run.

Two further facts. At 8 cards, 6-only play is better than before: at 1,500 deals the MLP loses 6.42 and the trees 6.02, where the old encoding lost about 8 to 10, and more 6-card training makes 8-card play better still for both. At 5 cards both models lose about 5 to 6 points, where the old encoding lost about 3.4 to 4.3, so these inputs are a weaker description of 5-card positions. On the trees, which passed the check, 5-card loss rose from 250 to 4,000 deals (+0.55, interval above 0). That comparison was not pre-registered.
