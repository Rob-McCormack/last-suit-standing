# Size-shift follow-ups

Three follow-ups to [size_shift.md](size_shift.md). **The predictions below were written and committed before any of these runs.** The git history shows the commit time.

All runs reuse `size_shift.py`: the same exact solver for any card count, the same features, policy and test deals (200 per size), and the same metric (points lost per game against perfect play, both seats). "Better than greedy" and "worse than greedy" are decided by a paired bootstrap 95% CI on the per-deal difference from the greedy player on the same deals. If the CI includes zero, the result is "no difference", and a prediction of better or worse is then scored as **not confirmed**.

## 1. Nearest-size test

**Hypothesis** (from the mixed-training test). A network extends to an unseen card count using the nearest card count it was trained on. If the nearest size has the same parity as the test size, it transfers well (better than greedy); if not, badly (worse than greedy).

**Design.** Two new mixes, 1,500 deals per size:

| Network | Test size | Nearest training size | Parity of nearest | Prediction |
| --- | --- | --- | --- | --- |
| Mix 4+5 | 6 | 5 | opposite | **worse** than greedy |
| Mix 4+5 | 7 | 5 | same | **better** than greedy |
| Mix 4+5 | 8 | 5 | opposite | **worse** than greedy |
| Mix 5+6 | 4 | 5 | opposite | **worse** than greedy |
| Mix 5+6 | 7 | 6 | opposite | **worse** than greedy |
| Mix 5+6 | 8 | 6 | same | **better** than greedy |

The Mix 4+5 rows are the risky ones. The network has trained on an even size (4) yet is predicted to fail on the even sizes 6 and 8, and to succeed on the odd size 7.

**Decision rule.** The hypothesis is **supported** if at least 5 of the 6 predictions are confirmed, and **rejected** if 3 or fewer are.

## 2. Positive control: tell the network who makes the last offer

**Question.** Do the networks fail because they cannot *work out* the last-offerer rule, or because it cannot be learned at all from this data?

**Design.** Add one input: whether Player 1 will make the last offer. Train on 6+7 cards, so this input varies during training (in 6-card games alone it would be constant). Compare with the same mix without the input: from the mixed-training test, 4.44 points lost on 5 cards and 15.36 on 8.

**Predictions.** With the explicit input, the network:
- **(a)** is better than greedy on 8 cards,
- **(b)** is better than greedy on 5 cards,
- **(c)** loses fewer points on 8 cards than the same mix without the input (paired CI excludes zero).

If (a)–(c) hold, the rule is learnable once stated, and the failure is in working it out. If they fail, the problem is deeper than one missing input.

## 3. Robustness

**Design.** Repeat the two headline comparisons with two more MLP seeds (seeds 2 and 3), and with a different kind of model: gradient-boosted trees (scikit-learn `HistGradientBoostingRegressor`, seeds 0 and 1).

**Predictions**, for each model type separately:
- **(d)** A network trained on 6 cards loses more extra points on 7 cards than on 8, compared with greedy on the same deals.
- **(e)** A network trained on 6+7 cards loses more points on 8 cards than one trained on 6 cards alone.

## Results

Added after the runs. Raw output is in `size_shift_followup_nearest.txt`, `size_shift_followup_control.txt` and `size_shift_followup_robust.txt`.

### Summary

| Part | Outcome |
| --- | --- |
| 1. Nearest-size test | **4 of 6 confirmed: undecided** by the pre-registered rule. All four predicted failures happened; both predicted successes did not. |
| 2. Positive control | **2 of 3 confirmed.** Telling the network who makes the last offer helps a lot, but it is still worse than greedy on 8 cards. |
| 3. Robustness | **4 of 4 confirmed.** Both headline findings hold in two new MLP seeds *and* in a different kind of model, gradient-boosted trees. |

### 1. Nearest-size test: not supported

Points lost per game against perfect play (seeds 0 and 1):

| Network | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- |
| Mix 4+5 | *0.45* | *1.42* | 8.77 | 9.51 | 17.98 |
| Mix 5+6 | 2.83 | *1.57* | *2.64* | 11.12 | 10.25 |
| Greedy | 1.29 | 2.90 | 4.41 | 7.26 | 9.88 |

| Prediction | Observed (network − greedy, 95% CI) | |
| --- | --- | --- |
| Mix 4+5 worse than greedy on 6 | worse, +4.36 [+3.65, +5.16] | confirmed |
| Mix 4+5 **better** than greedy on 7 | **worse**, +2.24 [+1.37, +3.06] | not confirmed |
| Mix 4+5 worse than greedy on 8 | worse, +8.11 [+6.86, +9.35] | confirmed |
| Mix 5+6 worse than greedy on 4 | worse, +1.55 [+1.25, +1.84] | confirmed |
| Mix 5+6 worse than greedy on 7 | worse, +3.86 [+2.86, +4.86] | confirmed |
| Mix 5+6 **better** than greedy on 8 | no difference, +0.37 [−0.57, +1.30] | not confirmed |

The four confirmations are all predicted failures, and failing on an unseen size turns out to be common. The two predictions that carried real information, the successes, both failed, including the risky one that mix 4+5 would handle 7 cards well. **The nearest-size explanation is not supported.** No network in this test beat greedy play on a card count it had not trained on.

**Caveat (found later).** This test is confounded by the fixed input slots. Mix 4+5 has never seen the 6th and 7th row slots or the 3rd stack slots filled, so on 7 cards it reads several inputs it never trained on, while the greedy player sees every card. The failed "better than greedy" predictions may partly reflect that. [slot_free.md](slot_free.md) tests the slot artifact for the main findings.

### 2. Positive control: stating the rule helps, but does not fix it

| Network | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- |
| Mix 6+7 with last-offerer input | 1.93 | **2.42** | *2.42* | *4.26* | **11.39** |
| Mix 6+7 without it (rerun) | 3.11 | 4.44 | *2.58* | *4.26* | 15.36 |
| Greedy | 1.29 | 2.90 | 4.41 | 7.26 | 9.88 |

The rerun without the input reproduces the earlier mixed-training numbers exactly.

- (a) Better than greedy on 8 cards: **not confirmed.** It is still worse, by +1.51 [+0.70, +2.33].
- (b) Better than greedy on 5 cards: **confirmed**, −0.49 [−0.96, −0.05].
- (c) Fewer points lost on 8 cards than without the input: **confirmed**, −3.97 [−4.98, −2.96].

Being told who makes the last offer removes much of the damage: it cuts the 8-card loss by a quarter, and on 5 cards the network goes from worse than greedy to better. But it does not close the gap on 8 cards. The last-offerer rule explains part of the failure, not all of it.

### 3. Robustness: the headline findings hold

| Model | Trained on | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- |
| MLP, seeds 2–3 | 6 only | 1.50 | 4.32 | *2.46* | 10.57 | 10.48 |
| MLP, seeds 2–3 | Mix 6+7 | 2.69 | 4.68 | *2.51* | *4.32* | 15.67 |
| Trees, seeds 0–1 | 6 only | 1.10 | 3.32 | *2.04* | 8.53 | 7.63 |
| Trees, seeds 0–1 | Mix 6+7 | 1.26 | 3.39 | *2.36* | *3.22* | 12.98 |
| Greedy | — | 1.29 | 2.90 | 4.41 | 7.26 | 9.88 |

- (d) The 6-card model does worse on 7 cards than on 8, relative to greedy. **Confirmed** for both: MLP +3.31 vs +0.60, trees +1.26 vs −2.25.
- (e) Adding 7-card training hurts 8-card play. **Confirmed** for both: MLP +5.20 [+4.08, +6.29] (per seed +5.17, +5.23); trees +5.35 [+4.33, +6.35] (per seed +5.35, +5.36).

One earlier detail does not hold up. In the original seeds, the 6-card MLP beat greedy on 8 cards (8.70 vs 9.88); in seeds 2–3 it does not (10.48). The comparison *between* 7 and 8 cards holds; "better than greedy on 8" is seed-dependent.

## What we now know

**Solid**, replicated across four MLP seeds and a second model type:
1. A model trained on 6 cards does worse on 7 cards (opposite parity, nearest size) than on 8 cards (same parity).
2. **Adding 7-card training data makes 8-card play worse by about 5 points per game.** The effect is nearly identical in neural networks and gradient-boosted trees, so it is not a quirk of one kind of model.

**Ruled out or weakened:**
- A simple shortcut, where the card count stands in for whose turn it is, does not explain it (mixed-training test).
- "Extrapolate from the nearest training size" is not supported (part 1).

**Partly explained:** stating the last-offerer rule as an input recovers a large part of the loss, but not all of it (part 2). With the slot-free encoding, replacing the network's offers with perfect play still cuts the loss by more than replacing its choices, on 6-only at 7 cards and on mix 6+7 at 8 cards ([component_repair.md](component_repair.md)).

**Open question:** why does extra training on 7 cards hurt play on 8 cards, so consistently, across different kinds of model?
