#!/usr/bin/env python3
"""Wild-card variant: how does it change perfect play?

Variant rule (pairing only): the last card sets trump as usual. Any stack card
with the SAME RANK as the last card is wild for pairing: it may pair with any
one other card in its owner's stack. Card values do not change (a wild 3 still
scores 3, doubled if trump). Each card is in at most one pair. If both players
hold a matching card, both are wild.

Example: the last card is 3H, so hearts are trump. A player holding 3C and 9S
may pair them: both are doubled, 3 + 9 -> 6 + 18.

Exact solver comparison against the official rules over random deals.
Standard library only.
Usage: python3 experiments/wild_rule.py [--deals 3000] [--seed 1]
"""
import argparse
import os
import random
import sys
from functools import lru_cache
from itertools import combinations

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import solver as S  # noqa: E402

R = S.OFFICIAL


def stack_value(stack, table, wild):
    """Trump doubles; the best single pair doubles both its cards. With at most
    three cards per stack, at most one pair is possible."""
    trump, wild_rank = table[1], table[0]
    vals = [r * 2 if s == trump else r for r, s in stack]
    best_pair = 0
    for i, j in combinations(range(len(stack)), 2):
        ri, rj = stack[i][0], stack[j][0]
        if ri == rj or (wild and (ri == wild_rank or rj == wild_rank)):
            best_pair = max(best_pair, vals[i] + vals[j])
    return sum(vals) + best_pair


@lru_cache(maxsize=None)
def swap(s1, s2, table, wild):
    """Official swap rule: Player 2 swaps first, Player 1 may make any exchange."""
    def m(a, b):
        return stack_value(a, table, wild) - stack_value(b, table, wild)

    def opts(a, b):
        yield a, b
        for _, na, nb in S._exchanges(a, b):
            yield na, nb
    return min(max(m(u1, u2) for u1, u2 in opts(t1, t2)) for t2, t1 in opts(s2, s1))


@lru_cache(maxsize=None)
def value(row, s1, s2, off, wild):
    if len(row) == 1:
        last = row[0]
        return (last[0] if off == 2 else -last[0]) + swap(s1, s2, last, wild)
    best = None
    for i in range(len(row) - 1):
        vs = [d + value(r, a, b, 3 - off, wild) for _, d, r, a, b in S._choices(row, s1, s2, off, i)]
        cv = min(vs) if off == 1 else max(vs)
        best = cv if best is None else (max(best, cv) if off == 1 else min(best, cv))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deals", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rng = random.Random(0)
    for _ in range(100):  # without the wild rule this is the official game
        d = tuple(rng.sample(S.DECK, 6))
        assert value(d, (), (), 1, False) == S.solve(d)
        value.cache_clear(); S.value.cache_clear()
    print("sanity: without the wild rule, values match solver.py on 100 deals")

    rng = random.Random(args.seed)
    deals = [tuple(rng.sample(S.DECK, 6)) for _ in range(args.deals)]
    sg = lambda x: (x > 0) - (x < 0)
    base, wild, two_of_rank = [], [], 0
    for d in deals:
        base.append(value(d, (), (), 1, False))
        wild.append(value(d, (), (), 1, True))
        value.cache_clear()
        ranks = [r for r, _ in d]
        two_of_rank += len(set(ranks)) < len(ranks)  # a wild needs two cards of one rank in the deal
    n = len(deals)
    wb, ww = sum(v > 0 for v in base) / n, sum(v > 0 for v in wild) / n
    tb, tw = sum(v == 0 for v in base) / n, sum(v == 0 for v in wild) / n
    print(f"\n{n} random deals (seed {args.seed})")
    print(f"deals where a wild is possible at all (some rank appears twice among the 6 cards): {two_of_rank / n:.1%}")
    print(f"Player 1 wins: official {wb:.1%}, wild {ww:.1%} (change {100 * (ww - wb):+.1f} points)")
    print(f"ties:          official {tb:.1%}, wild {tw:.1%}")
    print(f"deals whose perfect-play value changes: {sum(a != b for a, b in zip(base, wild)) / n:.1%}")
    print(f"deals whose result (win/tie/loss) changes: {sum(sg(a) != sg(b) for a, b in zip(base, wild)) / n:.1%}")
    up = sum(b > a for a, b in zip(base, wild)) / n
    down = sum(b < a for a, b in zip(base, wild)) / n
    print(f"value moves toward Player 1 in {up:.1%} of deals, toward Player 2 in {down:.1%}")
    print(f"mean margin (P1 - P2): official {sum(base) / n:+.2f}, wild {sum(wild) / n:+.2f}")


if __name__ == "__main__":
    main()
