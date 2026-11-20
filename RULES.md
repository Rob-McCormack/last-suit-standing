# Last Suit Standing

*One Card to Rule Them All.*

A two-player card game of offers, trump and swaps. A game takes about five minutes.

## What you need

- A 40-card deck: ace through 10 in four suits. Remove the face cards from a standard deck.
- Poker chips to keep score: white chips worth 1 point and red chips worth 5. Pen and paper works too.

**A card's value is the number on it** (ace = 1, up to 10). Points scored during play are kept as poker chips.

## Rules

1. **Set up.** Shuffle and deal six cards face up in a row. Both players start with zero chips. Player 1 offers first.

2. **Offer and choose.** On your turn, offer two neighboring cards in the row. Your opponent takes one of them into their face-up stack and takes chips equal to its value. You take chips equal to the value of the card left behind. That card stays in the row and can score again. Close the gap. Your opponent now offers. Repeat until one card remains.

3. **Set trump.** The last card's suit is trump. The last offerer takes chips equal to its value once more. Leave the card on the table. It belongs to neither stack and its scoring is complete.

4. **Swap.** The last chooser acts first. Each player gets one turn. On your turn you may pass, or exchange one card from your stack for one card from your opponent's stack. The second player may make any exchange, including swapping back the two cards the first player just exchanged. Chips stay with their owners.

5. **Score your stack.** Add the value of each card in your stack to your chips. For this final scoring, double a card's value if it is trump, and double it again if it belongs to a pair (two cards with the same number). Each card can belong to only one pair. With three of the same number, choose which two to pair.

**Highest total wins. Equal totals are a tie.**

## Clarifications

These follow from the rules above and are written out to remove doubt.

- **Turn order.** With six cards there are five offers. Player 1 makes offers 1, 3 and 5; Player 2 makes offers 2 and 4. So Player 1 is always the last offerer and Player 2 is always the last chooser. Player 1 ends with 2 cards in their stack and Player 2 with 3. Player 2 swaps first and Player 1 swaps last.
- **Neighboring** means next to each other in the row as it stands now, after earlier gaps have closed.
- **Pairs.** Both cards of a pair are doubled. A trump card in a pair is worth four times its value.
- **The last card** pays its value to the last offerer twice: once as the card left behind on the final offer, and once more under rule 3. Its suit sets trump, which can help either player. It never joins a stack.
- **Swapping does not move chips.** Only the cards in the stacks change hands.

## Worked example

Deal, left to right: 3♠ 4♦ 10♥ 2♣ 7♠ A♥. This line of play is perfect for both sides, as computed by `solver.py`.

| Offer | Offerer | Pair offered | Taken by chooser | Scored by offerer | Chips (P1–P2) |
| --- | --- | --- | --- | --- | --- |
| 1 | P1 | 3♠ 4♦ | P2 takes 3♠ | 4 | 4–3 |
| 2 | P2 | 4♦ 10♥ | P1 takes 10♥ | 4 | 14–7 |
| 3 | P1 | 4♦ 2♣ | P2 takes 4♦ | 2 | 16–11 |
| 4 | P2 | 2♣ 7♠ | P1 takes 7♠ | 2 | 23–13 |
| 5 | P1 | 2♣ A♥ | P2 takes A♥ | 2 | 25–14 |

The last card is 2♣, so clubs are trump and Player 1 collects 2 more chips (27–14). With perfect play, both players pass at the swap. No stack card is a club and there are no pairs. Player 1 scores 27 + 7 + 10 = 44. Player 2 scores 14 + 1 + 3 + 4 = 22. Player 1 wins by 22.
