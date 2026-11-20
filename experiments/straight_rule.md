# Straight-rule experiment

**Question.** Does adding a straight, a scoring rule that is trivial for the solver but relational and shared across both players, make the game much harder to *learn* while changing perfect play only a little?

This tests the designer's observation that some complicated rules barely change the solver's results but make the game much harder to learn. **The rule, predictions and decision rules below were committed before any runs.**

## The rule (variant, not part of the standard game)

At the final scoring:

1. A **straight** is three consecutive ranks among the five stack cards (both players') plus the last card on the table. Ace is low only (A-2-3); there is no wraparound.
2. Only the **highest** straight counts. In a run such as 4-5-6-7, the straight is 5-6-7.
3. The player holding the straight's **top card** scores a bonus equal to the three ranks added up (5-6-7 scores 18). The cards still score normally.
4. Nobody scores the straight if the top card is the table card, or if both players hold a card of the top rank.
5. A card may count in both a pair and a straight.

Everything else is unchanged: six cards, the official swap rule.

## Design

**Solver side** (exact, 3,000 random deals, seed 1): Player 1's win rate under each rule, the share of deals whose perfect-play value changes, the share whose result (win/tie/loss) changes, and the share of deals in which a straight is possible at all.

**Learning side.** The same models and data sizes as the earlier experiments: MLP and gradient-boosted trees, 3 seeds each, trained on 2,000 deals with exact values under the rule in force. They are tested on 400 held-out deals under the same rule, against perfect play, in both seats. Because the straight adds points, raw point losses are not comparable between rules. The **primary metric is therefore scale-free**:

- **Optimal-move rate:** the share of the model's decisions that lose nothing against perfect play.

Secondary measures: points lost relative to the greedy player under the same rule (model loss ÷ greedy loss), and points lost per game.

## Pre-registered predictions

- **(S1) Solver.** Player 1's win rate changes by **less than 5 percentage points**.
- **(S2) Solver.** The result changes in **fewer than 15%** of deals.
- **(L1) Learning, primary.** For **both** model types, the optimal-move rate is **lower** under the straight rule, by at least 3 percentage points. Each seed's difference is reported.
- **(L2) Learning.** For both model types, the ratio of model loss to greedy loss is higher under the straight rule.

**My own expectation.** I expect L1 and L2 to hold. I am less sure about S1 and S2: straights should be common in six cards, so the solver-level effect may be larger than the designer's intuition suggests. If S1 or S2 fails, the solver side is not "small"; the learning side is then judged against it.

## Results

Added after the run. Raw output is in `straight_rule_results.txt`. Sanity check: without the straight, the values match `solver.py` on 100 deals.

**Headline.** This straight rule is **not** a small change for the solver. It makes the game noticeably more lopsided: Player 1 wins 72.1% instead of 61.4%. It also makes the game somewhat harder to learn: the models' gap to perfect play, measured against greedy, grows by about a quarter to two-fifths. So it is a big rule for the solver and a moderate one for learning, not the "small for the solver, large for learning" pattern that was predicted.

### Part 1: solver (3,000 deals)

| Measure | Official | With straight |
| --- | --- | --- |
| A straight is possible at all (three consecutive ranks among the 6 cards) | — | 48.9% of deals |
| Player 1 wins under perfect play | 61.4% | **72.1%** (+10.7 points) |
| Deals whose perfect-play value changes | — | 43.7% |
| Deals whose result (win/tie/loss) changes | — | **17.9%** |
| Mean absolute margin | 5.40 | 8.60 |

- (S1) Player 1's win rate changes by less than 5 points: **not confirmed** (+10.7).
- (S2) The result changes in fewer than 15% of deals: **not confirmed** (17.9%).

Why Player 1 gains is an interpretation, not tested: Player 1 swaps last and may make any exchange, so Player 1 is usually the one able to take the straight's top card.

### Part 2: learning (400 test deals, 3 seeds)

| Model | Rule | Optimal-move rate (per seed) | Points lost | Model ÷ greedy loss |
| --- | --- | --- | --- | --- |
| MLP | official | 85.9% (85.9, 86.2, 85.6) | 2.46 | 0.56 |
| MLP | straight | 83.0% (83.2, 82.2, 83.6) | 3.51 | **0.69** |
| Trees | official | 87.6% (87.4, 87.6, 87.7) | 1.89 | 0.43 |
| Trees | straight | 84.3% (84.0, 84.5, 84.4) | 3.06 | **0.60** |
| Greedy | official | 80.9% | 4.42 | — |
| Greedy | straight | 80.6% | 5.06 | — |

- (L1) The optimal-move rate drops by at least 3 points: **trees confirmed** (+3.3; per seed +3.4, +3.1, +3.3). **MLP not confirmed** (+2.9, just under the threshold; per seed +2.7, +3.9, +2.0).
- (L2) The model ÷ greedy loss ratio is higher under the straight: **confirmed for both** (MLP 0.56 → 0.69; trees 0.43 → 0.60).

**An informative detail:** the greedy player's optimal-move rate hardly moves (80.9% → 80.6%), while both learned models drop about 3 points. The straight makes the decisions harder *for learned models in particular*, not just harder to score.

## What this shows

1. **"Complicated rule, little effect on the solver" did not hold for this rule.** Straights are possible in about half of all deals, so the rule changes perfect play a lot. It also shifts the balance further toward Player 1.
2. **The rule does make learning harder, but moderately.** Learned models fall further behind perfect play, relative to a greedy baseline whose own accuracy stays flat.
3. **For the game design:** this version of the straight makes Player 1's edge larger (72% wins). If straights are wanted, a version that favors the player *without* the last swap might rebalance the game instead.

The designer's original observation may still hold for other rules, especially rare ones. Testing it needs a rule that rarely applies, so that perfect play hardly changes, which this straight is not.

## Limits

One straight rule (one of several reasonable designs), 3 seeds, 400 test deals, and models graded against a perfect opponent only. The 3-point threshold in L1 was a pre-registered cut-off; the MLP missed it by 0.1 points.

## Reproduce

```bash
pip install numpy scikit-learn
python3 experiments/straight_rule.py --solver-deals 3000 --train-deals 2000 --test-deals 400 --seeds 0,1,2
```

This takes about 11 minutes on 2 CPU cores.
