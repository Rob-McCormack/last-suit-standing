# Checkpoint control

**Question.** Finding 1 says more 6-card training makes a neural net worse at other card counts. The net in that experiment stops when a 6-card validation split stops improving. This run asks whether that stopping rule is the cause. The same nets are trained for a fixed number of passes, with no early stopping, and compared at the same pass.

**The predictions below were written and committed before any runs.** The git history shows the commit time.

## Design

Same solver, deals, rollouts, suit relabelling, inputs, one-step policy, and metric as `scaling.py`. The metric is points lost per game against perfect play, both seats. Test deals are the same 200 held-out deals per size. This run tests 6, 7, and 8 cards only.

- **Data.** Nested prefixes of the scaling stream: 250, 1,000, and 4,000 six-card deals. Seeds 0, 1, 2, and 3.
- **Neural net.** The same MLP (two hidden layers, 256 and 128; Adam; learning rate 0.001; batch size 256). Early stopping is off, and the training-loss stopper is off, so every net runs for exactly 200 passes over its training rows. It trains on all of those rows. The scaling run held out 10% of the rows for stopping; this one does not.
- **Stages.** The net is tested after 10, 20, 40, 80, 160, and 200 passes. A matched stage means the same pass count.
- **Selection check.** A fixed set of 60 validation deals at each of 6, 7, and 8 cards is held out of training and out of the test set. At each stage the net's value error (mean squared error) is measured on those deals. For each seed and training size, two chosen stages are recorded: the lowest error on 6-card validation deals, and the lowest average of the three sizes' errors. Ties take the earliest stage.
- **Trees.** The same gradient-boosted trees, 500 iterations, early stopping off, on the same three prefixes and four seeds. They are tested only at the end. This is a check that they still differ from the net, not a second copy of the stage rule.
- **Intervals.** The deal comparison is the paired difference per test deal, averaged over seeds, with a bootstrap 95% CI over deals. The four seed values are reported separately and are not folded into that interval. "Rises" means the CI is entirely above 0 and at least 3 of the 4 seeds are positive. "Improves" means the CI is entirely below 0.

## Pre-registered predictions

| | Prediction | Call |
| --- | --- | --- |
| a | At pass 200, 6-card loss falls from 250 to 4,000 deals | Yes |
| b | At pass 200, 7-card loss rises from 250 to 4,000 deals | Yes |
| c | 7-card loss rises from 250 to 4,000 deals at every stage | No |
| d | The mixed-size validation choice still has higher 7-card loss at 4,000 than at 250 | No |
| e | Trees at 500 iterations: 7-card loss does not rise from 250 to 4,000 | Yes |
| f | The 6-card validation choice still has higher 7-card loss at 4,000 than at 250 | Yes |

The reason for (c) and (d): early passes have not fit the 6-card games yet, so more data should not hurt there. A check that can see 7-card value error can pick one of those earlier nets. The reason for (b) and (f): by the end, and whenever the chosen net is the one that fits 6 cards best, more 6-card data still entrenches the 6-card rule. Trees did not show that pattern when they were allowed to stop early, and (e) says they still do not.

## How finding 1 is judged

- If (a) is not observed, the neural-net run is uninformative, and finding 1 is left as it stands.
- Finding 1 stands unchanged only if 7-card loss rises at every stage.
- If it rises at pass 200 but not at every stage, the README claim is revised to the stages where it rises: more data hurts 7-card play only after the net has had those passes, not from the start of training.
- If it rises at no stage, the data-amount claim is withdrawn. The scaling result is then attributed to the stopping rule.
- If (b) is observed and (d) is not, add that a validation check which includes other card counts would not have picked the fragile net. If (f) is observed, add that a 6-card check still would have.
- If (e) is not observed, the trees no longer differ: they also get worse at 7 cards when early stopping is off.

The 1,000-deal row and the 8-card column are reported. They do not decide the finding.

## Results

Added after the run. Raw output is in `checkpoint_results.txt`. The run took 2,562 seconds. Seeds 0–2 have the same six-card position counts as `scaling_results.txt` (252,726; 253,005; 252,987).

**Six and seven cards, 4,000 deals minus 250.** Negative means play improved. "Rises" means the deal interval is entirely above 0 and at least 3 of the 4 seeds are positive.

| Pass | 6 cards | Per seed | 7 cards | Per seed |
| --- | --- | --- | --- | --- |
| 10 | −0.92 [−1.16, −0.68] improves | −0.87, −0.89, −0.95, −0.96 | **+2.66 [+1.94, +3.42] rises** | +4.05, +2.35, +3.12, +1.14 |
| 20 | −0.99 [−1.28, −0.71] improves | −0.52, −1.08, −1.24, −1.12 | **+1.96 [+1.32, +2.64] rises** | +3.18, +2.01, +1.88, +0.76 |
| 40 | −1.13 [−1.42, −0.84] improves | −0.77, −1.18, −1.60, −0.97 | **+0.95 [+0.37, +1.53] rises** | +1.71, +1.27, +1.44, −0.64 |
| 80 | −1.38 [−1.70, −1.08] improves | −1.14, −1.29, −1.88, −1.22 | **+1.04 [+0.44, +1.64] rises** | +1.74, +2.42, −0.43, +0.42 |
| 160 | −1.59 [−1.92, −1.26] improves | −1.34, −1.64, −1.60, −1.76 | **+1.01 [+0.36, +1.67] rises** | +2.13, +1.29, +0.10, +0.53 |
| 200 | −1.79 [−2.16, −1.46] improves | −1.64, −2.11, −1.86, −1.54 | **+0.85 [+0.21, +1.51] rises** | +1.80, +0.49, +0.69, +0.42 |

At passes 40 and 80, one seed went the other way on 7 cards. The interval is still above 0, and three seeds are positive, so those stages count as a rise. The 1,000-deal loss sits between the 250 and 4,000 losses at every pass except pass 80, where 1,000 (11.85) is slightly worse than 4,000 (11.54).

**Chosen passes.** The pass with the lowest validation error, 6-card error first and mixed-size error second:

| Seed | 250 deals | 1,000 | 4,000 |
| --- | --- | --- | --- |
| 0 | 80 / 10 | 20 / 10 | 80 / 10 |
| 1 | 40 / 10 | 10 / 10 | 40 / 10 |
| 2 | 20 / 10 | 20 / 10 | 40 / 20 |
| 3 | 20 / 10 | 20 / 10 | 10 / 10 |

On 7 cards, 4,000 minus 250: the 6-card choice is +1.49 [+0.88, +2.07], seeds +1.74, +1.27, +2.71, +0.22. The mixed-size choice is +2.64 [+1.93, +3.38], seeds +4.05, +2.35, +3.04, +1.14. Mixed-size error picks an early pass, and the early pass is where the 7-card gap is largest.

**Trees**, 500 iterations, no early stopping. From 250 to 4,000 deals: 6 cards −1.01 [−1.26, −0.76]; 7 cards +0.51 [+0.02, +1.03], seeds +1.00, +0.31, +0.36, +0.39; 8 cards −1.24 [−1.73, −0.74].

Eight cards for the net is not part of the decision. It rises at pass 10 (+0.95), is flat at 20, 40, and 80, and improves at 160 and 200 (−0.81, then −1.01).

### Scored predictions

| | Observed | Call |
| --- | --- | --- |
| a | Yes | Yes |
| b | Yes | Yes |
| c | Yes | No |
| d | Yes | No |
| e | No | Yes |
| f | Yes | Yes |

(a), (b), and (f) hit. (c) missed: the 7-card loss rises at every pass, including pass 10. (d) missed: the mixed-size choice still rises, and by more than the final pass. (e) missed: the trees rise too.

### What this means under the decision rules

(a) was observed, so the run is informative. Seven-card loss rises at every matched pass, so **finding 1 stands unchanged.** The stopping rule is not the cause. At pass 10 the net trained on 4,000 deals is already better at 6 cards and worse at 7, by 2.66 points.

The mixed-size check does not remove the damage, so there is nothing to add about a broader validation check avoiding the fragile net. The 6-card check still selects a net that is worse at 7 cards.

The tree statement does not hold. With early stopping off, the trees also get worse at 7 cards, by 0.51 points, in all four seeds. That rise is smaller than the net's. In the earlier experiment, where the trees were allowed to stop early, their 7-card change was flat.

A pass is one trip through the training rows. With batch size 256, the 4,000-deal net takes about sixteen times as many Adam updates per pass as the 250-deal net (about 988 against about 62). The decision above compares matched passes. At matched update counts that rise does not survive: more deals lowered 7-card loss, and training the same 4,000-deal net longer still raised it ([matched_steps.md](matched_steps.md)).
