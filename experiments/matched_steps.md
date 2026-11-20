# Matched Adam updates

**Question.** Finding 1, as tested in `checkpoint.md`, says that at the same training pass, more 6-card deals make 7-card play worse. A pass is one trip through the rows. With a batch size of 256, the 4,000-deal net takes about sixteen times as many Adam updates per pass as the 250-deal net. This run asks whether the 7-card rise is still there when the update counts match.

**The predictions below were written before this run.** The direction was suggested by lining up the already published pass table in `checkpoint_results.txt` at similar update counts. Those lined-up means are not a result. This run's paired intervals are the result. The git history shows the commit time.

## Design

Same solver, deals, rollouts, suit relabelling, inputs, one-step policy, and metric as `checkpoint.py`. The metric is points lost per game against perfect play, both seats. Test deals are the same 200 held-out deals per size, at 6, 7, and 8 cards. Training excludes those deals and the same 60 validation deals per size that `checkpoint.py` excluded, so the row counts should match that run. This run does not use the validation scores.

- **Data.** Nested prefixes of the scaling stream: 250, 1,000, and 4,000 six-card deals. Seeds 0, 1, 2, and 3.
- **Neural net.** The same MLP as `checkpoint.py`: two hidden layers, 256 and 128; Adam; learning rate 0.001; batch size 256. Early stopping is off, and the training-loss stopper is off. One call to `partial_fit` is one pass. An update is one Adam step on one batch, including a final short batch, so a pass contains `ceil(rows / 256)` updates.
- **Budgets.** 10,000, 20,000, and 40,000 updates. For each training set the recorded model is the first pass whose update count is at least the budget. The actual count is printed. The larger set can overshoot a budget by almost one pass. At 10,000 updates that overshoot is under about 9 percent for the 4,000-deal net, and well under that for the smaller nets.
- **Intervals.** The deal comparison is the paired difference per test deal, averaged over seeds, with a bootstrap 95% CI over deals. The four seed values are reported separately. "Improves" means the CI is entirely below 0 and at least 3 of the 4 seeds are negative. "Rises" means the CI is entirely above 0 and at least 3 of the 4 seeds are positive.

## Pre-registered predictions

| | Prediction | Call |
| --- | --- | --- |
| a | At the 10,000-update budget, 7-card loss improves from 250 deals to 4,000 | Yes |
| b | At the 20,000-update budget, 7-card loss improves from 1,000 deals to 4,000 | Yes |
| c | At the 40,000-update budget, 7-card loss improves from 1,000 deals to 4,000 | Yes |
| d | On the 4,000-deal nets, 7-card loss rises from the 10,000-update budget to the 40,000-update budget | Yes |

(a), (b), and (c) say that once the number of updates is matched, more deals help 7-card play. (d) says that training the same 4,000-deal net for longer still hurts 7-card play.

## How finding 1 is judged

- (a) decides. The 6-card and 8-card columns are reported and do not decide.
- If (a) improves, and neither (b) nor (c) rises, the claim that more deals cause the 7-card damage is withdrawn. The README then says the matched-pass rise did not survive matched updates: at the same update count, more 6-card deals lowered 7-card loss. If (d) also rises, add that training the 4,000-deal net longer still made 7-card play worse.
- If (a) improves but (b) or (c) rises, the direction depends on the budget. No single rewrite. The data-amount sentence stays undecided.
- If (a) rises, the data-amount claim stands at matched updates, and the README says so.
- If (a) is neither a rise nor an improvement, this test does not decide. The README keeps the note that the checkpoint comparison did not separate deals from updates.

## Results

Added after the run. Raw output is in `matched_steps_results.txt`. The run took 489 seconds. The six-card position counts match `checkpoint_results.txt` (252,726; 253,005; 252,987; 252,357).

Update counts are the first pass at or above the budget. At the 10,000 budget the 4,000-deal nets were recorded at about 10,868 updates, and the 250-deal nets at about 10,037.

**Seven cards, mean of seeds 0–3.** Negative means the larger training set, or the later budget, lost fewer points.

| Comparison | Change | Per seed |
| --- | --- | --- |
| a 4,000 minus 250, about 10,000 updates | **−0.99 [−1.69, −0.27] improves** | +1.15, −1.94, −1.49, −1.69 |
| b 4,000 minus 1,000, about 20,000 updates | **−0.93 [−1.63, −0.25] improves** | −0.27, −0.10, −1.93, −1.44 |
| c 4,000 minus 1,000, about 40,000 updates | **−0.93 [−1.54, −0.31] improves** | +0.92, −0.78, −1.52, −2.35 |
| d 4,000 deals, 40,000 minus 10,000 updates | **+0.61 [+0.17, +1.06] rises** | +0.24, +1.43, +0.38, +0.40 |

On (a) and (c), seed 0 went the other way. Three seeds are negative, and the interval is below 0, so both still count as an improvement.

Absolute 7-card losses at about 10,000 updates were 11.45, 10.51, and 10.46 for 250, 1,000, and 4,000 deals. At about 40,000 updates they were 12.70, 12.01, and 11.07.

Six and eight cards do not decide. At about 10,000 updates, 4,000 minus 250 was −1.68 [−1.99, −1.40] on 6 cards and −3.85 [−4.47, −3.23] on 8 cards.

### Scored predictions

| | Observed | Call |
| --- | --- | --- |
| a | Yes | Yes |
| b | Yes | Yes |
| c | Yes | Yes |
| d | Yes | Yes |

All four hit.

### What this means under the decision rules

(a) improves, and neither (b) nor (c) rises. The claim that more deals cause the 7-card damage is withdrawn. At a matched update count, more 6-card deals lowered 7-card loss. (d) rises, so the README also says that training the 4,000-deal net longer still made 7-card play worse. The matched-pass rise in `checkpoint.md` stands as a result about passes. It is not a result about the number of deals.
