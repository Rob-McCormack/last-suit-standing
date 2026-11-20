# Relation to Collapse3

Last Suit Standing and [Collapse3](https://github.com/Rob-McCormack/collapse-3) ([overview](https://github.com/Rob-McCormack/collapse3-beat-the-best)) belong to one research program: small games solved exactly, used to measure where AI players and AI evaluations go wrong.

Collapse3 has the larger body of results. Its findings all come from one game and siblings built on the same engine. Last Suit Standing is a **second, independent game** that tests whether those findings hold when the game is built differently, and it adds a theme of its own: **rule changes**.

## How the games differ

| | Collapse3 | Last Suit Standing |
| --- | --- | --- |
| Start | One fixed opening position per size | A random deal; billions of distinct games |
| Scoring | Win / draw / loss (plus a win-type ladder) | Points margin |
| Length | Many moves; the full game is deep | Five offers and a two-step swap per deal |
| Moves | Placement, removal, gravity | Offer-and-choose, trump, swaps |
| Exact tools | Enumeration up to (5,5); root solve at full size | Every deal solved in milliseconds |

## What replicates

| Collapse3 finding | In Last Suit Standing | Strength |
| --- | --- | --- |
| **17: The strongest player is not the strongest tester.** Testing only against perfect opponents misses weaknesses. | Testing against the perfect solver found only 59–70% of the points a worst-case opponent extracts. In about 1 game in 6 it showed no loss where the worst case found one. ([rule_shift_exploit.md](experiments/rule_shift_exploit.md)) | **Replicates, in milder form** |
| **17: One deliberate mistake (tremble budget m = 1) exposes everything.** | One mistake per game recovers the full worst case in 88–96% of games, not always. This matches Collapse3's own caveat that finer-grained scoring can need a second unit, and Last Suit Standing scores points. | **Mostly replicates; the difference is explained** |
| **16: Generalization is not robustness.** A network ~98% optimal on unseen positions is still a certified forced loss (12 of 12). | Trained networks are force-worsened in 15–33% of games by the worst-case opponent. No all-or-nothing result, because each deal is short and a mistake often costs points rather than the result. | **Same direction, much milder** |
| **9: A value audit is silent between equally good moves.** | Tied best openings differ after a rule change in 15.2% of deals, by up to 42 points. Choosing among equal moves is invisible under the current rules but matters under the next. ([hidden_gap.md](experiments/hidden_gap.md)) | **Analogue, extended to rule changes** |
| **14(iii): A learned strategy carried across a size boundary is forced to lose.** | A network trained on 6 cards fails worst on 7 cards, the nearest size but with the opposite parity, and plays worse than untrained greedy play on both odd sizes. Same-parity transfer beats opposite-parity transfer in 10 of 10 comparisons. ([size_shift.md](experiments/size_shift.md)) | **Replicates, with a parity boundary** |
| **10: Elo ranks an exploitable rulebook above the perfect player.** | Not replicated: in a six-player round robin, Elo ranked every player in its true order ([elo.md](experiments/elo.md)). This supports Collapse3's explanation that the inversion comes from its drawn root, because Last Suit Standing has none. Elo still could not separate two models that differ by 20% in points lost. | **Does not replicate; the contrast supports Collapse3's mechanism** |
| **6: Seat is a confound in win rate.** | Player 1 wins 61.4% under perfect play. A one-sentence change to the swap rule flips the edge to Player 2. | **Replicates, with a rule-change twist** |

## What is new here

- **Rule changes as the main theme.** Collapse3 notes rule sensitivity only as a caution: one tiebreak ruling moved which feature mattered. Last Suit Standing measures it directly. A network trained on the official rules loses 4.4 extra points per game under the undo rule against the worst-case opponent. The cost is uneven: the reverse shift costs 1.9. ([rule_shift.md](experiments/rule_shift.md), [rule_shift_exploit.md](experiments/rule_shift_exploit.md))
- **More training can increase fragility.** A neural network trained on more 6-card games improved at 6 cards and got steadily worse at other card counts, while gradient-boosted trees did not ([scaling.md](experiments/scaling.md)). Held-out tests at the training size point the wrong way, which echoes Collapse3's theme that standard scoreboards can endorse the weaker agent.
- **Rates across many starting positions.** Because each deal is its own small game, results are percentages over hundreds of deals with confidence intervals, rather than certificates from one opening.

## What Last Suit Standing does not test

Partial observation and aliasing floors (Collapse3 Findings 4, 5, 7, 15), scaling with board size (11–13), self-play (14) and sandbagging (9, first half) have no counterpart here yet.

## How to present it

Collapse3 has the headline results. Last Suit Standing is supporting evidence that the evaluation failures are not quirks of one game. It shows they appear, more mildly, in a game with chance and points scoring. It also opens a new question about how trained players and evaluations cope when the rules themselves change.
