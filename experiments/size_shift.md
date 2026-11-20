# Size-shift experiment

**Question.** A network learns the game with 6 cards. Does what it learned carry over to games with more or fewer cards?

**Answer.** It depends on whether the new card count is odd or even, more than on how close it is. The 6-card network does far worse on 7 cards than on 8: it loses **7.0 extra points per game** at 7 cards against 3.3 at 8, compared with networks trained on those sizes. At both odd sizes it plays **worse than a simple greedy player**, so its training actively hurts. Across every size tested, a network trained two cards away with the same parity beat one trained only one card away with the opposite parity (**10 of 10** comparisons).

## Why card count matters

With N cards there are N − 1 offers, so the parity of N decides who moves last. With an even N, Player 1 makes the last offer, collects the last card's bonus and swaps last. With an odd N, Player 2 does all three. The rules are otherwise identical; the last chooser always swaps first.

Under perfect play this changes the game completely. Exact values on the 200 test deals per size used below:

| Cards | Last offerer | P1 wins | P1 ties |
| --- | --- | --- | --- |
| 4 | P1 | 49% | 14% |
| 5 | P2 | 2% | 1% |
| 6 | P1 | 57% | 13% |
| 7 | P2 | 2% | 0% |
| 8 | P1 | 84% | 3% |

With an odd number of cards, Player 2 almost always wins: the last mover's advantage is not offset, because both stacks are the same size. A network that learned 6-card play has learned a game where Player 1 has the last word.

## Setup

- **Exact solver for any N.** The rules are the same as `RULES.md`, except that the last chooser swaps first whichever player that is. (`solver.py` itself assumes six cards and always has Player 2 swap first. This script carries its own general version, which matches `solver.py` on 6-card deals.)
- **Networks.** scikit-learn MLPs (hidden layers 256 and 128), trained from scratch on exact values. One network per training size, from 4 to 8 cards, so every test size has a control trained on that size. Training uses 1,500 deals per size per seed. Positions come from rollouts mixing perfect and random moves, and suits are relabelled for augmentation.
- **Inputs** are padded to 8 row slots and 4 stack slots per player. A 6-card network has never seen some of these slots filled. A 7-card game fills two of them (the 7th row slot and Player 1's 3rd stack slot); an 8-card game fills four (both extra row slots, Player 1's 3rd and Player 2's 4th stack slot). So the encoding artifact is *larger* at 8 cards, where the network transfers better. (An earlier version of this file said both sizes were affected equally; that was wrong. [slot_free.md](slot_free.md) tests the artifact directly.)
- **Policy.** One-step lookahead using the network's value of the next position. The network makes every offer and choice for its seat; swaps are played perfectly.
- **Test.** 200 held-out deals per size, both seats, against perfect play. 2 seeds.
- **Metric.** Points lost per game compared with perfect play. Larger games involve more points, so compare each column with its own control, not across columns.

## Results

Points lost per game compared with perfect play, both seats averaged, mean over 2 seeds. Rows show the training size and columns the test size; the diagonal is the control.

| Trained on ↓ / tested on → | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- |
| 4 cards | **0.37** | 7.72 | 6.75 | 15.18 | 14.47 |
| 5 cards | 2.60 | **1.72** | 8.58 | 8.82 | 15.68 |
| 6 cards | 1.34 | 4.31 | **2.42** | 11.05 | 8.70 |
| 7 cards | 2.52 | 2.32 | 6.98 | **4.03** | 13.45 |
| 8 cards | 1.78 | 3.56 | 4.09 | 9.41 | **5.43** |
| Greedy (no training) | 1.29 | 2.90 | 4.41 | 7.26 | 9.88 |

**The 6-card network's transfer penalty**, meaning extra points lost compared with a network trained on the test size (95% CI over deals):

| Test size | Parity | Extra points lost | Per seed | Compared with greedy |
| --- | --- | --- | --- | --- |
| 4 | same | +0.97 [0.74, 1.20] | +0.91, +1.02 | about equal (1.34 vs 1.29) |
| 5 | **opposite** | +2.59 [2.05, 3.14] | +2.48, +2.69 | **worse** (4.31 vs 2.90) |
| 7 | **opposite** | **+7.03** [6.09, 7.95] | +7.09, +6.96 | **worse** (11.05 vs 7.26) |
| 8 | same | +3.27 [2.54, 4.03] | +2.56, +3.98 | better (8.70 vs 9.88) |

**Parity beats distance.** For every test size, compare networks trained two cards away (same parity) with networks trained one card away (opposite parity). The same-parity network loses fewer points in **10 of 10** comparisons. Two of the ten are close: 6.75 vs 6.98 at 6 cards, and 8.82 vs 9.41 at 7.

The share of games ending in a worse result (win/tie/loss) than perfect play is in `size_shift_results.txt`.

## What this shows

- **Transfer follows the game's structure, not its size.** Seven cards is the nearest size to six, yet it is where the 6-card network fails worst. Eight cards is further away but structurally the same game, and transfer is much better.
- **Learning can be worse than not learning.** On odd card counts the trained network plays worse than a greedy player with no training at all: what it learned is actively misleading there. On even counts it matches or beats greedy.
- **The failure is invisible from inside the training size.** Nothing in 6-card play hints that one more card flips who wins. This is the same pattern as Collapse3's Finding 14(iii), where a self-play agent carried an opening across a size boundary and was forced to lose. Here it is a parity boundary, crossed in both directions, and measured across thousands of games.

## Testing an explanation: the shortcut hypothesis fails

**Hypothesis.** In 6-card games, the number of cards left fully determines whose turn it is: Player 1 offers with 6, 4 or 2 cards left, and Player 2 with 5 or 3. A network trained only on 6 cards cannot tell which of the two matters. If it keyed on the card count, odd sizes, which reverse the pairing, would mislead it.

**Test** (`size_shift_mixed.py`). Two networks were trained on the same amount of data. **Mix 6+7** breaks the link, because the two sizes pair card count and turn in opposite ways. **Mix 6+8** keeps it, since both sizes are even. The prediction was that only mix 6+7 would transfer to both parities. Same test deals, metric and seeds as above.

| Network ↓ / tested on → | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- |
| 6 cards only (from above) | 1.34 | 4.31 | 2.42 | 11.05 | 8.70 |
| Mix 6+7 | 3.11 | 4.44 | 2.58 | *4.26* | **15.36** |
| Mix 6+8 | 0.91 | 4.66 | 2.30 | 9.55 | *5.36* |
| Greedy (no training) | 1.29 | 2.90 | 4.41 | 7.26 | 9.88 |

*Italics: a size the network was trained on.* Mix 6+8 minus mix 6+7, with 95% CIs: 4 cards −2.20 [−2.50, −1.88]; 5 cards +0.22 [−0.25, +0.69]; 7 cards +5.28 [+4.38, +6.17]; 8 cards −10.01 [−11.19, −8.83]. Both seeds agree in direction.

**The hypothesis is falsified.** Breaking the link did not produce a network that handles both parities:

- On 5 cards, mix 6+7 is no better than the 6-card network (4.44 vs 4.31), and still worse than greedy.
- On 8 cards, **adding 7-card training made things much worse**: 15.36 points lost, against 8.70 for the 6-card network and 9.88 for untrained greedy play. More varied training data hurt transfer.

> **Update.** The hypothesis below was then tested with pre-registered predictions and was **not supported**. Both main findings did replicate across more seeds and a second model type. See [size_shift_followup.md](size_shift_followup.md).

**What fits instead (a new hypothesis, not yet tested): the networks extrapolate from the nearest size they know.** They do not learn the rule that whoever makes the last offer gets the bonus and the final swap. Read the table this way:

- The 6-card network treats 5 and 7 like 6, which is wrong, and 4 and 8 like 6, which is right.
- Mix 6+7 treats 8 like 7, which is wrong and is the worst cell in the table, and 5 like 6, also wrong.
- Mix 6+8 treats 4 like 6 (right), 5 like 6 (wrong), and 7 as a blend of 6 and 8 (wrong).

Every cell is consistent with this, but it was read off after the fact. Testing it properly needs a design chosen in advance, for example training on 5+6 and predicting 7 (nearest size 6, wrong parity) against 4 (nearest 5, wrong parity).

**What this means for "memorizing or learning?"** The networks are neither memorizing nor learning the rule. They generalize to unseen deals of the sizes they trained on, and seem to extend to new sizes by **resemblance**, not by understanding *why* one player is favored. Showing them both parities was not enough to teach the reason.

## Limits

One architecture, 2 seeds, 200 test deals per size, and graded against a perfect opponent only (the weakest tester; see [rule_shift_exploit.md](rule_shift_exploit.md)). The networks are small and use one-step lookahead. The inputs include the number of cards left and who is offering, so the parity is in principle visible to the network; it simply did not learn what it means outside its training size.

## Reproduce

```bash
pip install numpy scikit-learn
python3 experiments/size_shift.py --train-deals 1500 --test-deals 200 --seeds 2
```

This takes about 25 minutes on 2 CPU cores. The 8-card solves are the slow part. The mixed-training test is `python3 experiments/size_shift_mixed.py --train-deals 1500 --test-deals 200 --seeds 2`, also about 25 minutes, with output in `size_shift_mixed_results.txt`. For a quick check, `--train-deals 80 --test-deals 40 --seeds 1` runs in about 40 seconds. The output of this run is in `size_shift_results.txt`.
