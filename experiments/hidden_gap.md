# Hidden-gap experiment

**Question.** In some deals, two or more opening offers are exactly tied for best under the official rules, so no test under those rules can tell them apart. How often do those tied openings differ under the undo swap rule, and by how much?

**Answer.** Often, and by a lot. In **15.2%** of deals, the tied best openings have different values under the undo rule. The average hidden gap is **1.19 points per deal**, and **7.8 points** among the deals where it occurs. Both exceed the pre-registered thresholds. The result is exact: no learning or approximation is involved.

This idea was proposed by Kimi. The code here is an independent implementation.

## Why it matters

A player can be perfect under the current rules and still be fragile when a rule changes, because perfection under the current rules does not determine *which* of several equally good moves it picks. In machine learning this is known as underspecification. Here it can be measured exactly: how often it happens, and how much it costs.

## Definitions

For each deal *d* and opening offer *a*:

- **S(a)**: the exact final margin (Player 1 minus Player 2) after opening *a*, with perfect play under the official rules.
- **U(a)**: the same under the undo rule. The game after the opening is re-solved under that rule, so any loss comes from the opening alone.
- **A**: the set of openings tied for the highest S.
- **Hidden gap G**: the best U in A minus the worst U in A. If G > 0, the official rules cannot tell a robust opening from a fragile one.

**Pre-registered prediction** (from Kimi's proposal): at least 5% of deals are fragile (G > 0), and the mean G is at least 0.25 points.

## Results

5,000 random deals, seed 2026:

| Measure | Predicted | Result | 95% CI |
| --- | --- | --- | --- |
| Deals with tied best openings | — | 18.0% | — |
| Fragile deals (G > 0) | at least 5% | **15.2%** | 14.2% to 16.2% (Wilson) |
| Mean hidden gap | at least 0.25 points | **1.19 points** | 1.08 to 1.30 (bootstrap) |
| Mean gap among fragile deals | — | 7.8 points (max 42) | — |

Most deals with tied openings (15.2 of 18.0 percentage points) hide a real difference.

For comparison, an opening that is optimal under the official rules loses 2.24 points per deal on average under the undo rule, whichever way ties are broken. Much of that comes from deals with a single best opening that is simply wrong under the new rule. The hidden gap is the part that **no** official-rule information could help with: it depends only on how a tie happens to be broken.

### Example: the largest gap

Deal: 6♣ 3♥ 2♦ 10♣ 7♠ 10♦

| Opening | Official rules | Undo rule |
| --- | --- | --- |
| 6♣–3♥ | **+1** (tied best) | −16 |
| 3♥–2♦ | **+1** (tied best) | −4 |
| 2♦–10♣ | −4 | −51 |
| 10♣–7♠ | **+1** (tied best) | −46 |
| 7♠–10♦ | **+1** (tied best) | −16 |

Four openings are exactly equal under the official rules. Under the undo rule they range from −4 to −46, a 42-point difference that nothing in the official game could reveal.

## Not yet tested

Kimi's secondary prediction is that a network trained only on official-rule values would tend to break these ties toward the fragile option. That needs a trained tie-breaker and has not been run here.

## Limits

This covers opening offers only, so it is a lower bound on how much ambiguity exists across the whole game. Values are computed under margin play (see `solver.md`); under win/tie/loss grading, more moves would count as tied.

## Reproduce

```bash
python3 experiments/hidden_gap.py --deals 5000 --seed 2026
```

Standard library only; about 70 seconds. The output of this run is in `hidden_gap_results.txt`.
