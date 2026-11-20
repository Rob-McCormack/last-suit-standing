#!/usr/bin/env python3
"""Basic facts about perfect and greedy play (the README's pilot numbers).

Measures, over random deals under the official rules:
  1. how often the chooser's best move is to take the LOWER of the two cards
  2. how much the swap phase matters: how often Player 2's first swap changes
     the result, and how often both players pass
  3. how a greedy player fares against a perfect opponent, in each seat

Greedy player: as offerer, picks the pair that maximizes its immediate chip
margin, assuming the chooser takes the higher card; as chooser, takes the higher
card. It never looks at stacks, trump or the opponent's actual reply. (The
greedy baseline in rule_shift.py is a simpler variant that maximizes the
offerer's own chips; the two are reported separately.)
The swap phase is played perfectly by both sides throughout.

Standard library only.
Usage: python3 experiments/basic_stats.py [--deals 3000] [--seed 1]
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import solver as S  # noqa: E402

R = S.OFFICIAL


def perfect_choice(opts, off):
    pick = min if off == 1 else max
    return pick(opts, key=lambda o: o[1] + S.value(o[2], o[3], o[4], 3 - off, R))


def perfect_offer(row, s1, s2, off):
    pick = max if off == 1 else min
    return pick(range(len(row) - 1), key=lambda j: S.offer_value(row, s1, s2, off, j, R))


def greedy_offer(row, s1, s2, off):
    # assume the chooser takes the higher card; maximize the offerer's immediate
    # chip margin (lower card scored minus higher card given away)
    return max(range(len(row) - 1),
               key=lambda i: min(row[i][0], row[i + 1][0]) - max(row[i][0], row[i + 1][0]))


def greedy_choice(opts, off):
    return max(opts, key=lambda o: o[0][0])


def play(deal, greedy_seat=None, stats=None):
    """Final margin (P1 - P2). greedy_seat plays greedily; the rest perfectly."""
    row, s1, s2, off, margin = tuple(deal), (), (), 1, 0
    while len(row) > 1:
        i = greedy_offer(row, s1, s2, off) if off == greedy_seat else perfect_offer(row, s1, s2, off)
        opts = list(S._choices(row, s1, s2, off, i))
        chooser = 3 - off
        o = greedy_choice(opts, off) if chooser == greedy_seat else perfect_choice(opts, off)
        if stats is not None and greedy_seat is None:
            vals = [x[1] + S.value(x[2], x[3], x[4], 3 - off, R) for x in opts]
            stats["all_choices"] += 1
            if vals[0] != vals[1]:  # a real choice: count whether the best card is the lower one
                stats["choices"] += 1
                best = opts[vals.index(min(vals) if off == 1 else max(vals))]
                other = opts[1] if best is opts[0] else opts[0]
                stats["take_lower"] += best[0][0] < other[0][0]
        margin += o[1]
        _, _, row, s1, s2 = o
        off = 3 - off
    last = row[0]
    if stats is not None and greedy_seat is None:
        full, p2_move, p1_move = S.swap_phase(s1, s2, last[1], R)
        # value if Player 2 could not swap first: Player 1 alone picks its best swap
        def m(a, b):
            return S.stack_value(a, last[1], R) - S.stack_value(b, last[1], R)
        p1_only = max([m(s1, s2)] + [m(na, nb) for _, na, nb in S._exchanges(s1, s2)])
        stats["p2_swap_matters"] += full != p1_only
        stats["both_pass"] += p2_move is None and p1_move is None
    return margin + S.value(row, s1, s2, off, R)


def rates(margins, seat):
    n = len(margins)
    win = sum((m > 0) if seat == 1 else (m < 0) for m in margins) / n
    tie = sum(m == 0 for m in margins) / n
    return win, tie, 1 - win - tie


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deals", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    deals = [rng.sample(S.DECK, 6) for _ in range(args.deals)]

    stats = dict(all_choices=0, choices=0, take_lower=0, p2_swap_matters=0, both_pass=0)
    perfect, greedy1, greedy2 = [], [], []
    for d in deals:
        v = play(d, None, stats)
        assert v == S.solve(d, R)
        perfect.append(v)
        greedy1.append(play(d, greedy_seat=1))
        greedy2.append(play(d, greedy_seat=2))
        S.value.cache_clear()

    n = args.deals
    print(f"{n} random deals (seed {args.seed}), official rules\n")
    print(f"Chooser's best card is the LOWER one: {stats['take_lower']}/{stats['choices']} "
          f"real choices = {stats['take_lower'] / stats['choices']:.1%} "
          f"({stats['take_lower'] / stats['all_choices']:.1%} of all {stats['all_choices']} choices)")
    print(f"Player 2's first swap changes the result: {stats['p2_swap_matters'] / n:.1%} of deals")
    print(f"Both players pass at the swap: {stats['both_pass'] / n:.1%} of deals\n")
    for label, ms, seat in (("Perfect P1 vs perfect P2", perfect, 1),
                            ("Greedy P1 vs perfect P2", greedy1, 1),
                            ("Perfect P1 vs greedy P2", greedy2, 1),
                            ("Greedy P2 vs perfect P1 (P2's view)", greedy2, 2),
                            ("Perfect P2 vs perfect P1 (P2's view)", perfect, 2)):
        w, t, l = rates(ms, seat)
        print(f"{label:<38} win {w:6.1%}  tie {t:5.1%}  loss {l:6.1%}")


if __name__ == "__main__":
    main()
