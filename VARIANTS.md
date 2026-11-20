# Rule variants

These variants are used for research and game-design testing. They are **not** part of the standard game, which is described in [RULES.md](RULES.md).

The first two are built into `solver.py` (`--swap undo`, `--trump half`). The others have their own scripts in `experiments/`.

- **Undo-only swap:** if the first player makes an exchange, the second may only pass or swap those same two cards back. If the first player passes, the second may make any exchange. This one change flips which seat is favored (see [rule_shift.md](experiments/rule_shift.md)). (A stricter reading, where the second player may never start a new exchange, makes the swap phase irrelevant under perfect play: either player can always keep the original stacks.)
- **Half trump:** trump cards are worth half their rank, rounded down, instead of double.
- **Straight (tested, not adopted):** the holder of the top card of the highest three-card straight, across both stacks and the table card, scores the ranks' sum. Under perfect play it raises Player 1's win rate from 61% to 72% (see [straight_rule.md](experiments/straight_rule.md)).
- **Wild match (tested, under consideration):** any stack card with the same rank as the last card is wild for pairing. It may pair with any one other card in its owner's stack; card values do not change. Under perfect play, Player 1's win rate falls from 61% to 54%, a more balanced game (see `experiments/wild_rule.py` and `wild_rule_results.txt`).
- **Jacks wild (tested, not adopted):** a 44-card deck with four jacks; a jack is worth 0 chips during the offers and becomes any rank and suit at the final scoring. Under perfect play, Player 1's win rate rises to 70% overall, and to 79% in deals that contain a jack (see `experiments/jacks_rule.py`).
