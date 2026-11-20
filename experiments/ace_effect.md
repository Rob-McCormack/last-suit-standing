# Ace-suit test

**Question.** Changing the suit of an ace swings perfect play more than changing the suit of a 10, because low cards tend to be left behind, survive to the end and set trump (idea from ChatGPT). Do trained models notice this?

**Answer.** No. Both models react to the ace's suit and the 10's suit by the same amount, even though the true effect of the ace is 70% larger. They capture **none** of the ace's extra importance.

**Prediction, stated before running:** the models underrate the ace. Their estimated change, relative to the true change, is smaller for the ace than for the 10. **Confirmed for both models.**

## Setup

- **400 matched pairs** (seed 3). Each pair is two 6-card deals identical except for the suit of one target card, either an ace or a 10, at a random position.
- **True change:** the difference in perfect-play value between the two deals, from the exact solver.
- **Estimated change:** the difference in a trained model's value estimate for the two opening positions. The models are the MLP and gradient-boosted trees from the earlier experiments: 6 cards, 1,500 training deals, seed 0.

## Results

| Target card | Pairs | True change (points) | Winner flips | MLP estimate | Trees estimate | MLP ÷ true | Trees ÷ true |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Ace | 185 | **4.30** | 37.8% | 2.14 | 0.48 | **0.50** | **0.11** |
| 10 | 215 | 2.53 | 25.6% | 2.19 | 0.53 | 0.87 | 0.21 |

- In truth, the ace's suit matters 1.78 points more than the 10's. Both models estimate it as 0.06 points *less*, so they capture none of the gap.
- The MLP recovers 87% of the true effect for the 10 but only 50% for the ace. The difference is −0.37 (95% CI −0.60 to −0.18).
- The trees barely react to suit changes at all (11% and 21%). They still underrate the ace relatively: −0.10 (95% CI −0.15 to −0.06).

## What this shows

The models treat a card's suit as mattering about equally whatever its rank. They miss that the ace's suit is far more decisive because the ace usually becomes the last card and sets trump. This is a small, concrete case of trained models overlooking a high-leverage detail whose immediate value looks small.

## Limits

One training seed per model and 400 pairs. The test looks at the models' value estimates for the opening position, not at their moves; the models choose moves by looking one step ahead, so their play may partly compensate.

## Reproduce

```bash
python3 experiments/ace_effect.py --pairs 400 --seed 3
```

This takes about 90 seconds. The output is in `ace_effect_results.txt`.
