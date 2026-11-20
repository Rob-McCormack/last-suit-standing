# Three projects, one research program

Last Suit Standing is one of three projects that share a method: **small games solved exactly, used to find where AI and its evaluations go wrong.** Because every position has a known correct answer, every claim can be checked exactly rather than estimated.

Each project studies a different kind of failure.

| Project | Studies | In one line |
| --- | --- | --- |
| **PSDG** (Philosopher's Stone Dice Game) | What happens when a perfect plan is locked in and the opponent goes off-script | Frozen plans |
| **Collapse3** | How we measure AI, and how standard tests make weak AI look strong | Misleading measurement |
| **Last Suit Standing** | What happens when the rules or the game's size change | Change and transfer |

## PSDG: frozen plans

A player that commits to the solver's best line, instead of re-solving after each move, can lose to a blundering opponent. When play leaves the expected path, the locked-in plan may no longer be best.

- The finding does **not** depend on the simultaneous exchange. It holds under the sequential exchange too, where the second player sees the first player's choice. A static player loses 8.5% of games under the sequential exchange, against 6.9% under the simultaneous one.
- The game also lets a player set the state of the opponent's own pieces (the poisoned gift), which chess and Go never do.

The safety angle is deployment: a policy that is optimal on paper but frozen can fail when the world, or the opponent, leaves the path it planned for.

## Collapse3: misleading measurement

Repositories: [collapse-3](https://github.com/Rob-McCormack/collapse-3) (solver and findings) and [collapse3-beat-the-best](https://github.com/Rob-McCormack/collapse3-beat-the-best) (overview).

- **Scoreboards mislead.** Win rate, Elo and on-policy regret can rank an exploitable agent above the perfect one.
- **"Beat the best" certifies the least.** Testing against a perfect opponent never visits the positions where a weak player fails.
- **Generalization is not robustness.** Networks about 98% right on unseen positions were still certified forced losses, 12 of 12.
- **Partial observation has an exact price.** Hiding one feature from an agent creates an enumerated floor of unavoidable mistakes.

The safety angle is evaluation: we can build more capable systems faster than we can verify them.

## Last Suit Standing: change and transfer

See [README.md](README.md) and the `experiments` folder.

- **More training, more fragility.** A neural network trained on more 6-card games improved at 6 cards and got steadily worse at every other card count. At 7 cards it ended 65% worse than an untrained greedy player. Held-out tests at 6 cards would have picked it as the best model. Gradient-boosted trees did not do this. ([scaling.md](experiments/scaling.md))
- **One more card flips the game, and models do not notice.** Models trained on 6 cards fail worst on 7, the nearest size but the opposite parity. Adding 7-card training made 8-card play about 5 points worse, in two model types. ([size_shift.md](experiments/size_shift.md), [size_shift_followup.md](experiments/size_shift_followup.md))
- **Rule changes cost more than a perfect-opponent test shows.** The cost of a one-sentence rule change doubles when graded by the exact worst-case opponent. ([rule_shift_exploit.md](experiments/rule_shift_exploit.md))
- **Equally perfect choices are not equally safe.** In 15% of deals, tied best openings differ by up to 42 points after a rule change. ([hidden_gap.md](experiments/hidden_gap.md))

The safety angle is change: systems are trained and tested on one version of a task, then used on another.

## Where they overlap

The overlaps are replications, not duplicates, and they make the shared claims stronger.

| Idea | Where it appears |
| --- | --- |
| A frozen or memorized plan fails once the opponent leaves the expected line | PSDG (static commitment vs re-solving); Collapse3 Finding 3 (a memorized line went 0 wins, 17 draws, 15 losses against deviating opponents) |
| Testing against the best opponent misses weaknesses | Collapse3 Finding 17; Last Suit Standing, in milder form ([COLLAPSE3.md](COLLAPSE3.md)) |
| Good average performance hides exploitable mistakes | Collapse3 Finding 16; Last Suit Standing (worst-case testing of trained networks) |
| An opponent's deliberate blunder can be the strongest test | PSDG (a blundering opponent beats a static solver); Collapse3 (a one-mistake budget exposes weaknesses); Last Suit Standing (one deliberate mistake recovers most of the worst case) |

## How to present them

As one program with three angles:

> Small games, solved exactly, used to find where AI breaks: when a plan is frozen (PSDG), when the measurement misleads (Collapse3), and when the rules change (Last Suit Standing).

Each game confirms some of the others' findings in a structurally different setting. That is the program's main defence against the objection that any one result is a quirk of one game.

## Note on sources

The Last Suit Standing results are reproducible from this repository. The PSDG and Collapse3 summaries are taken from those projects' own documents and from earlier discussions; their own repositories are the authority for their numbers.
