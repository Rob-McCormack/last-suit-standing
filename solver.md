# How the solver works

`solver.py` computes perfect play for any six-card deal of Last Suit Standing. It uses only the Python standard library (Python 3.8 or later).

## Quick use

```bash
python3 solver.py 3S 4D 10H 2C 7S AH          # solve one deal and show perfect play
python3 solver.py                             # a random deal
python3 solver.py --deals 3000 --seed 1       # statistics over random deals
python3 solver.py 3S 4D 10H 2C 7S AH --check  # also confirm by brute force
```

Cards are a rank (`A` or `1` to `10`) followed by a suit (`S H D C`, or `♠ ♥ ♦ ♣`). Case does not matter.

For one deal it prints:

- the value of each of Player 1's five possible opening offers;
- one perfect line of play, with chips after every offer, the trump, both swap decisions and the final stacks;
- the result and margin.

## What "perfect play" means here

Each player maximizes their own final margin: Player 1 maximizes (P1 total − P2 total) and Player 2 minimizes it. The game is zero-sum with perfect information, so this is ordinary minimax.

**Margin is not the same as winning.** Under win/tie/loss scoring, a win by 1 is as good as a win by 10, but this solver prefers the win by 10. Because a bigger margin is never a worse result, the solver's moves are always win-optimal and the deal's result (win, tie or loss) is the same under both goals. The difference is in *which* moves count as correct: under win/tie/loss grading, several moves can be equally correct where the solver names only one. Keep this in mind before scoring a player's moves as right or wrong.

## Rule options

| Option | Values | Default |
| --- | --- | --- |
| `--swap` | `any`: the second swapper may make any exchange. `undo`: after an exchange, the second swapper may only pass or undo it; after a pass, they may make any exchange. | `any` (official, see RULES.md) |
| `--trump` | `double`: trump cards score ×2. `half`: ×½, rounded down. | `double` (official) |

The swap reading matters a great deal. Over 3,000 random deals (`--seed 1`):

| Swap reading | P1 wins | Ties | P1 loses | Mean margin |
| --- | --- | --- | --- | --- |
| `any` (official) | 61.4% | 8.2% | 30.5% | +3.27 |
| `undo` | 35.5% | 4.5% | 60.0% | −1.87 |

## How it computes

The game splits into two phases.

**Swap phase** (`swap_phase`). Once the last card is known, only the two stacks and the trump suit matter. Player 2 tries every option: pass, or each of its 3 × 2 = 6 exchanges. For each, Player 1 picks its best reply. Player 2 keeps the option whose best reply is worst for Player 1. That is at most 7 × 7 = 49 evaluations.

**Offer phase** (`value`). Each offer is a two-step move. The offerer picks one of the neighboring pairs, which is a max for Player 1 and a min for Player 2. The chooser then picks which card to take, the opposite of the offerer. Chips earned along the way add up, so the value of a position does not depend on the chips already scored. That lets results be cached by row, stacks, and who offers next. The full tree has at most 5·2 · 4·2 · 3·2 · 2·2 · 1·2 = 3,840 lines before the swap, so a deal solves in milliseconds.

**Scoring** (`stack_value`). Each card's rank is doubled (or halved) if it is trump. Within each rank, the two highest-valued cards form a pair and are doubled again. With three of a rank, the trump one is paired if there is one.

The rule options are part of every cache key, so switching rules in one Python session is safe.

## Using it from Python

```python
from solver import solve, opening_values, principal_line, brute_force, Rules, parse_card

deal = [parse_card(c) for c in "3S 4D 10H 2C 7S AH".split()]
solve(deal)                          # +22
solve(deal, Rules(swap="undo"))      # alternative swap reading
opening_values(deal)                 # value of each opening offer for P1
lines, p1_total, p2_total = principal_line(deal)
brute_force(deal)                    # slow, independent check
```

Cards are `(rank, suit)` tuples, for example `(10, "H")`. Call `value.cache_clear()` between batches if memory matters.

## How it was checked

- **Brute force.** `brute_force` is a separate, uncached search of the whole game tree, and `--check` runs it. It agreed on 75 of 75 checks: 25 deals under three rule settings.
- **Replay.** For each deal, the printed line of play is replayed and its final scores must equal the computed margin. The program asserts this every time.
- **Independent solvers.** On random deals the value matched ChatGPT's solver (200 of 200) and Kimi's under the undo reading (150 of 150). It also matched Grok's after two fixes: the opponent, not the offerer, picks the card taken, and any second swap is allowed.

## Card counts

`solver.py` handles an **even** number of cards, including the standard six. It assumes Player 1 makes the last offer and Player 2 swaps first, which is true only for even counts. `solve`, `opening_values` and `principal_line` refuse an odd count with an error. For any card count, use the solver in `experiments/size_shift.py`, where the last chooser swaps first. It matches `solver.py` on 6-card deals, and an independent brute-force search on 3-, 4-, 5- and 7-card deals.

## Assumptions

- Both cards of a pair are doubled. A trump card in a pair is worth four times its rank.
- Chips earned during play stay with their owner. Swaps move cards only.
- Ties between equally good moves are broken by order: the leftmost offer, then the first card of the pair. Only the printed line of play depends on this, never the value.
