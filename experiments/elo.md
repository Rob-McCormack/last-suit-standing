# Elo experiment

**Question.** Does an Elo rating from a round-robin tournament rank players in the same order as their true strength? Collapse3's Finding 10 showed Elo ranking an exploitable rulebook above the perfect player.

**The predictions below were committed before the run.**

## Players

| Player | Description |
| --- | --- |
| **Perfect** | The solver's minimax play |
| **Exploiter** | Plays the best response to the greedy player, assuming its opponent is greedy, whoever it actually faces |
| **MLP** | Neural network from `size_shift.py` (6 cards, 1,500 training deals, seed 0) |
| **Trees** | Gradient-boosted trees, same data |
| **Greedy** | Takes the higher card and offers the best immediate chip margin (as in `basic_stats.py`) |
| **Random** | Uniformly random offers and choices, seeded per game |

All players play the swap phase perfectly, as in the earlier experiments, so differences come from the offer-and-choose phase.

## Tournament

- A round robin on 300 random deals (seed 7). Every pair plays every deal **twice, once from each seat**, which removes the seat and deal advantage from the comparison.
- That is 15 pairs × 600 games = 9,000 games.
- **Elo** is fitted by maximum likelihood from win/tie/loss results (a tie counts as half a win), on the usual 400-point scale.
- **True strength** is measured exactly: mean points lost per game against the perfect player, both seats, on the same deals. Lower is stronger.

## Pre-registered predictions

1. **Perfect ranks first by Elo.** Unlike Collapse3's drawn game, perfect play here wins often against weak players, because ties are rare.
2. **A mid-table inversion:** the Exploiter's Elo is higher than that of at least one player that loses fewer points against perfect play.

A full Collapse3-style inversion, with Exploiter rated above Perfect, is predicted **not** to happen.

## Results

Added after the run. Raw output is in `elo_results.txt`. Sanity check: against Greedy, the Exploiter achieves exactly its computed best-response value on 30 of 30 deals checked.

**Headline.** Elo ranked all six players **in exactly their true order**. Collapse3's Elo inversion does not appear here.

| Elo rank | Player | Elo (95% CI) | Score rate | Points lost vs Perfect | True rank |
| --- | --- | --- | --- | --- | --- |
| 1 | Perfect | 1669 [1657, 1683] | 71.4% | 0.00 | 1 |
| 2 | Exploiter | 1623 [1613, 1634] | 65.0% | 1.90 | 2 |
| 3 | Trees | 1567 [1558, 1578] | 57.0% | 1.95 | 3 |
| 4 | MLP | 1565 [1555, 1577] | 56.6% | 2.40 | 4 |
| 5 | Greedy | 1475 [1461, 1489] | 43.7% | 4.46 | 5 |
| 6 | Random | 1100 [1077, 1120] | 6.3% | 15.08 | 6 |

Head-to-head score rates (row player against column player, both seats):

| | Perfect | Exploiter | MLP | Trees | Greedy | Random |
| --- | --- | --- | --- | --- | --- | --- |
| **Perfect** | — | 60.6% | 64.7% | 64.0% | 71.6% | 96.2% |
| **Exploiter** | 39.4% | — | 56.5% | 56.3% | 79.6% | 93.3% |
| **MLP** | 35.3% | 43.5% | — | 49.1% | 61.3% | 94.0% |
| **Trees** | 36.0% | 43.7% | 50.9% | — | 60.7% | 93.5% |
| **Greedy** | 28.4% | 20.4% | 38.8% | 39.3% | — | 91.6% |
| **Random** | 3.8% | 6.7% | 6.0% | 6.5% | 8.4% | — |

**Scoring the predictions**

1. Perfect ranks first by Elo: **confirmed**. Perfect leads the Exploiter by 46 Elo (95% CI 32 to 59).
2. The Exploiter is rated above an objectively stronger player: **not confirmed**. It ranks second by Elo *and* by true strength.

The second prediction failed for an unexpected reason. The Exploiter, built only to beat the greedy player, turned out to be the strongest non-perfect player overall: it loses only 1.90 points per game against perfect play, fewer than both trained models. So there was no stronger player for it to be wrongly ranked above.

One weakness of Elo did show: Trees and MLP are tied by Elo (1567 vs 1565, overlapping CIs), even though the trees lose about 20% fewer points against perfect play (1.95 vs 2.40). Elo, built on win/tie/loss, cannot separate players that win similar shares of games but by different margins.

## What this shows

- **In Last Suit Standing, Elo worked.** The order was correct, and the top gap is clear.
- **This supports Collapse3's explanation of its own result.** There, the inversion arises because the full game is a draw: the perfect player cannot earn more than draws against strong opponents and has no reason to run up wins against weak ones. Last Suit Standing has no drawn root (ties are about 8%), and perfect play wins often against weak players. Without that ceiling, the inversion does not appear. The contrast is evidence that the Collapse3 inversion comes from the drawn root, not from Elo in general.
- **Elo still loses information.** It cannot see margins, so two players with very different point losses can look identical.

## Limits

One tournament, 300 deals, six players and one exploiter design. A player built to exploit a *different* weakness might produce an inversion. The Exploiter is a full-depth search against a greedy opponent, and greedy play is often close to optimal, so it is in effect a strong search player rather than a narrow exploit; against Greedy it scores more than Perfect does (79.6% vs 71.6%), so a pool with more exploitable players could still produce an inversion. Strength is measured against the perfect opponent only, which is the weakest tester ([rule_shift_exploit.md](rule_shift_exploit.md)).

## Reproduce

```bash
pip install numpy scikit-learn
python3 experiments/elo.py --deals 300 --seed 7 --boot 300
```

This takes about 4 minutes on 2 CPU cores.
