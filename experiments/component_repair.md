# Component repair

**Question.** When a trained network plays badly at a new card count, is the damage in the offers, the choices, or both? The network's offer decisions, its choice decisions, or both are replaced with the solver's decisions. The rest of the game, including the opponent and the swaps, stays perfect.

## Design

The network is the MLP from `size_shift.py`. Seeds 0 and 1. 1,500 training deals per size. The same 200 held-out test deals per size as the earlier experiments. Two models: 6-only, and mix 6+7. Tested on 6, 7, and 8 cards. The metric is points lost per game against perfect play.

Four conditions. L means the network decides. S means the solver decides.

| | Offers | Choices |
| --- | --- | --- |
| LL | network | network |
| SL | solver | network |
| LS | network | solver |
| SS | solver | solver |

SS must lose nothing. The comparisons are paired over deals, averaged over seeds and seats, with a bootstrap 95% CI. "Helps" means the CI is entirely below 0.

- Offer repair: R_SL − R_LL.
- Choice repair: R_LS − R_LL.
- Offer repair helps more than choice repair when (R_SL − R_LL) − (R_LS − R_LL), which is R_SL − R_LS, has a CI entirely below 0.

The primary comparison is 6-only tested on 7 cards. The secondary is mix 6+7 tested on 8 cards. Six cards is the control.

## Old pre-registration

This block was written before the old-encoding run, on 2026-11-20 at 13:14:55 local time. Its sha256 is `97da11b9efa9bac6735fad26a950ec70a61d81a8d640b3fe85e5149a41c3be31`. It was **not** committed to git. That was Rob's request. The file was `component_repair_prereg.md`.

The predictions in it:

- **Kimi:** on 6-only tested on 7 cards, fixing only the offers hurts. R_SL − R_LL has a CI entirely above 0.
- **Claude:** on that same test, fixing only the offers helps on average (CI entirely below 0), but the repaired player loses more than the untouched network in at least 10% of (deal, seat) games.
- **Claude, unsure, stated anyway:** on that same test, choice repair helps (CI entirely below 0).

## Old-encoding results

These numbers are from that earlier run, on the old fixed-slot encoding. They were not produced by a committed pre-registration. Raw output was `component_repair_results.txt`. The LL row matches `size_shift.py`: 6-only on 7 cards is 11.05, and mix 6+7 on 8 cards is 15.36. SS is 0 everywhere.

Points lost per game, both seats, mean of seeds 0 and 1:

| Model, test | LL | SL | LS | SS |
| --- | --- | --- | --- | --- |
| 6-only, 6 | 2.42 | 0.94 | 1.76 | 0.00 |
| 6-only, 7 | 11.05 | 3.74 | 7.13 | 0.00 |
| 6-only, 8 | 8.70 | 1.92 | 6.79 | 0.00 |
| mix 6+7, 6 | 2.58 | 1.07 | 1.82 | 0.00 |
| mix 6+7, 7 | 4.26 | 1.26 | 3.18 | 0.00 |
| mix 6+7, 8 | 15.36 | 4.84 | 10.70 | 0.00 |

The two comparisons the pre-registration named:

| | Offer repair | Choice repair | Offer worse |
| --- | --- | --- | --- |
| 6-only on 7 | −7.32 [−8.33, −6.30] | −3.92 [−4.53, −3.33] | 9.4% of games |
| mix 6+7 on 8 | −10.52 [−11.67, −9.42] | −4.66 [−5.35, −3.98] | 11.6% of games |

Per seed, offer repair on 6-only at 7 cards was −7.30 and −7.33. Choice repair was −3.88 and −3.96. The interaction, R_SL + R_LS − R_LL − R_SS, was about zero on both of those tests (the intervals include 0).

**Scored against the old pre-registration**, for 6-only on 7 cards:

| Prediction | Result |
| --- | --- |
| Kimi: offer repair hurts | **Miss.** It helps, −7.32 [−8.33, −6.30]. |
| Claude: offer repair helps on average | **Hit.** |
| Claude: offer repair hurts in at least 10% of games | **Miss.** It is worse in 9.4% of games. |
| Claude, unsure: choice repair helps | **Hit.** −3.92 [−4.53, −3.33]. |

Offer repair helped more than choice repair on both named tests. Fixing the offers took 6-only on 7 cards from 11.05 to 3.74, and mix 6+7 on 8 cards from 15.36 to 4.84.

## Slot-free rerun: pre-registration

**Written before the slot-free run.** The same experiment, with `feats_slotfree` from `slot_free.py` instead of the old slot encoding. Seeds 0 and 1, 1,500 deals, the same test deals, the same four conditions. After that run, `--features plain` is repeated to check that the script reproduces the old LL numbers (11.05 and 15.36).

Each line below is predicted for **both** 6-only on 7 cards and mix 6+7 on 8 cards.

| | Prediction |
| --- | --- |
| a | Offer repair helps: R_SL − R_LL, CI entirely below 0 |
| b | Choice repair helps: R_LS − R_LL, CI entirely below 0 |
| c | Offer repair helps more than choice repair: R_SL − R_LS, CI entirely below 0 |

## Results

Added after the run. Raw output is in `component_repair_results.txt`. A tiny check (8 training deals, 2 test deals) was run first; those points are not used below.

**Plain-encoding check.** The same script was run again with the old encoding. It does **not** reproduce the earlier file exactly. 6-only on 7 cards is 11.12 here, against 11.05 before. Mix 6+7 on 8 cards is 15.69, against 15.36. This machine used scikit-learn 1.9.1. The earlier file remains the record of that run, and it was still not committed before that run. The two runs in this file used one script, so they can be compared with each other.

**Slot-free, both seats, mean of seeds 0 and 1.**

| Model, test | LL | SL | LS | SS |
| --- | --- | --- | --- | --- |
| 6-only, 7 | 11.40 | 5.01 | 7.12 | 0.00 |
| mix 6+7, 8 | 13.19 | 3.71 | 9.65 | 0.00 |

| | Offer repair | Choice repair | Offer minus choice |
| --- | --- | --- | --- |
| 6-only on 7 | −6.39 [−7.32, −5.49] | −4.28 [−4.83, −3.72] | −2.12 [−3.08, −1.18] |
| mix 6+7 on 8 | −9.48 [−10.51, −8.50] | −3.53 [−4.14, −2.94] | −5.95 [−6.91, −4.98] |

Per seed, offer repair is −6.28 and −6.51 on the first row, and −8.92 and −10.04 on the second. Choice repair is −4.21 and −4.34, then −3.73 and −3.33.

### Scored predictions

Each line was predicted for both tests. "Yes" means the interval is entirely below 0.

| | 6-only on 7 | mix 6+7 on 8 |
| --- | --- | --- |
| a Offer repair helps | Yes | Yes |
| b Choice repair helps | Yes | Yes |
| c Offer repair helps more | Yes | Yes |

All six hit. With the slots removed, fixing the offers still removes more of the loss than fixing the choices. On 7 cards the untouched network loses 11.40 and the offer-repaired one loses 5.01. On 8 cards those numbers are 13.19 and 3.71.
