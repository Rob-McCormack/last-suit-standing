# Rule-shift experiment

**Question.** A small neural network learns to judge positions under one swap rule. If that one sentence of the rules changes, how many extra points does it lose?

**Answer.** On the undo rule, a network trained on the official rules loses **2.2 more points per game** than an identical network trained on the undo rule. That is about 70% more loss (95% CI +1.8 to +2.7, consistent across 3 seeds). Going the other way costs only 0.8 points: the two directions are asymmetric.

> **Follow-up.** These results grade the networks only against the perfect solver, which is the weakest test. Against the exact worst-case opponent the rule-change penalty doubles, to +4.4 points per game. See [rule_shift_exploit.md](rule_shift_exploit.md).

## Setup

- **Networks.** Two scikit-learn MLPs (hidden layers 256 and 128), trained from scratch to predict the solver's exact value of a position. One is trained under the official swap rule (`any`), the other under `undo`. They share the same deals, features and seed.
- **Training data.** 2,000 random deals per seed. Positions come from rollouts that mix perfect and random moves, so the network also sees positions off the perfect line. Each position is also added with suits relabelled, since suits are interchangeable. That gives about 126,000 positions per network.
- **Policy.** One-step lookahead: for each offer or choice, pick the move with the best chips gained plus the network's value of the resulting position. The network makes every offer and choice for its seat. **Both sides play the swap phase perfectly**, so all loss comes from earlier moves made with a wrong picture of how the game ends.
- **Opponent.** The exact solver, under the rules actually in force.
- **Test.** 400 held-out deals; the network plays each deal once as Player 1 and once as Player 2.
- **Metric.** Points lost per game compared with perfect play (regret), averaged over both seats. It is never negative. A sanity check confirms that perfect play scores exactly 0 under both rules.
- **Flip deals.** 125 of the 400 test deals have a different best opening offer under the two rules.

## Results

Mean points lost per game against perfect play, both seats averaged, averaged over 3 seeds, with 95% bootstrap CIs over deals:

| Player | Tested on official (`any`) | Tested on `undo` |
| --- | --- | --- |
| Network trained on official | **2.46** [2.21, 2.72] | **5.35** [4.89, 5.84] |
| Network trained on undo | 3.29 [3.01, 3.59] | **3.14** [2.83, 3.46] |
| Greedy | 6.97 | 14.63 |
| Random | 15.04 | 23.17 |

Extra loss from playing under the rule a network was *not* trained on (paired on the same deals):

| Tested on | Extra loss (pts/game) | 95% CI | Per seed | Flip deals | Other deals |
| --- | --- | --- | --- | --- | --- |
| `undo` (network trained on official) | +2.21 | +1.79 to +2.65 | +1.91, +2.21, +2.52 | +2.72 | +1.98 |
| official (network trained on undo) | +0.83 | +0.60 to +1.07 | +0.73, +0.86, +0.91 | +1.27 | +0.64 |

Error on held-out positions under each network's own rule was 3.3–3.5 points for the official-rule networks and 3.9–4.2 for the undo-rule networks, so the undo game is somewhat harder to learn.

## What this shows

- A one-sentence rule change that only takes effect at the end of the game costs a trained network substantial points in the *opening and middle* of the game.
- The cost is uneven between the two directions: knowledge learned under the undo rule carries over to the official rule better than the reverse. Why is an open question.
- The general fact that trained models are brittle under rule changes is well known. This experiment adds a tiny, exact setting where that cost is priced in points against perfect play, with a matched control.

## Limits

One network architecture and size; 3 seeds; 400 test deals; one-step lookahead only. The policy assumes the opponent replies well, and the swap phase is perfect for both sides, so swap-phase errors are not measured. The greedy baseline here offers the pair whose lower card is highest; the other experiments use a slightly stronger greedy player that maximizes the immediate chip margin, so greedy numbers are not comparable across files.

## Reproduce

```bash
pip install numpy scikit-learn
python3 experiments/rule_shift.py --train-deals 2000 --test-deals 400 --seeds 3
```

This takes about 9 minutes on 2 CPU cores. The output of the run reported here is in `rule_shift_results.txt`. For a quick check, `--train-deals 200 --test-deals 60 --seeds 1` runs in about 15 seconds.
