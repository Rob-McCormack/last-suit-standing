#!/usr/bin/env python3
"""Jacks-wild variant: how does it change perfect play?

Variant rules (not part of the standard game):
  - The deck has 44 cards: ranks 1-10 in four suits plus four jacks.
  - During the offers a jack is worth 0 chips (taking or scoring one earns nothing).
  - If the last card is a jack, there is no trump and the last offerer collects 0 more.
  - At the final scoring each jack in a stack becomes whatever rank (1-10) and
    suit its owner likes best. It can pair with any card and can be trump.
    Each card is still in at most one pair.
Everything else follows RULES.md (official swap rule).

Exact solver over random deals, compared with the official 40-card game.
Standard library only.
Usage: python3 experiments/jacks_rule.py [--deals 3000] [--seed 1]
"""
import argparse
import os
import random
import sys
from functools import lru_cache
from itertools import combinations, product

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import solver as S  # noqa: E402

J = 11  # rank code for a jack
DECK44 = S.DECK + [(J, s) for s in S.SUITS]


def chips(card):
    return 0 if card[0] == J else card[0]


@lru_cache(maxsize=None)
def stack_value(stack, trump):
    """Best value of a stack; jacks take the rank and suit their owner likes best."""
    jacks = [i for i, c in enumerate(stack) if c[0] == J]
    fixed = [c for c in stack if c[0] != J]
    best = None
    for ranks in product(range(1, 11), repeat=len(jacks)):
        # a wild jack is best made trump if there is a trump
        cards = fixed + [(r, trump if trump else "S") for r in ranks]
        vals = [r * 2 if (trump and s == trump) else r for r, s in cards]
        pair = 0
        for i, j in combinations(range(len(cards)), 2):
            if cards[i][0] == cards[j][0]:
                pair = max(pair, vals[i] + vals[j])
        v = sum(vals) + pair
        best = v if best is None else max(best, v)
    return best


@lru_cache(maxsize=None)
def swap(s1, s2, trump):
    def m(a, b):
        return stack_value(tuple(sorted(a)), trump) - stack_value(tuple(sorted(b)), trump)

    def opts(a, b):
        yield a, b
        for _, na, nb in S._exchanges(a, b):
            yield na, nb
    return min(max(m(u1, u2) for u1, u2 in opts(t1, t2)) for t2, t1 in opts(s2, s1))


def choices(row, s1, s2, off, i):
    for k in (0, 1):
        take, stay = row[i + k], row[i + 1 - k]
        nr = row[: i + k] + row[i + k + 1:]
        if off == 1:
            yield chips(stay) - chips(take), nr, s1, tuple(sorted(s2 + (take,)))
        else:
            yield chips(take) - chips(stay), nr, tuple(sorted(s1 + (take,))), s2


@lru_cache(maxsize=None)
def value(row, s1, s2, off):
    if len(row) == 1:
        last = row[0]
        trump = None if last[0] == J else last[1]
        bonus = chips(last) if off == 2 else -chips(last)
        return bonus + swap(s1, s2, trump)
    best = None
    for i in range(len(row) - 1):
        vs = [d + value(r, a, b, 3 - off) for d, r, a, b in choices(row, s1, s2, off, i)]
        cv = min(vs) if off == 1 else max(vs)
        best = cv if best is None else (max(best, cv) if off == 1 else min(best, cv))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deals", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rng = random.Random(0)  # without jacks this is the official game
    for _ in range(100):
        d = tuple(rng.sample(S.DECK, 6))
        assert value(d, (), (), 1) == S.solve(d)
        value.cache_clear(); S.value.cache_clear()
    print("sanity: deals without jacks match solver.py on 100 deals")

    rng = random.Random(args.seed)
    vals, has_jack = [], []
    for _ in range(args.deals):
        d = tuple(rng.sample(DECK44, 6))
        vals.append(value(d, (), (), 1))
        value.cache_clear()
        has_jack.append(any(c[0] == J for c in d))
    n = len(vals)

    def rates(vs):
        k = len(vs)
        w, t = sum(v > 0 for v in vs) / k, sum(v == 0 for v in vs) / k
        return w, t, 1 - w - t, sum(vs) / k

    print(f"\n{n} random deals from the 44-card deck (seed {args.seed})")
    print(f"deals with at least one jack: {sum(has_jack) / n:.1%}")
    for label, vs in (("all deals", vals),
                      ("deals with a jack", [v for v, h in zip(vals, has_jack) if h]),
                      ("deals without a jack", [v for v, h in zip(vals, has_jack) if not h])):
        w, t, l, m = rates(vs)
        print(f"  {label:<22} P1 wins {w:6.1%}  ties {t:5.1%}  loses {l:6.1%}  mean margin {m:+6.2f}  ({len(vs)} deals)")
    print("official 40-card game for comparison: P1 wins 61.4%, ties 8.2%, loses 30.5%, mean +3.27")


if __name__ == "__main__":
    main()
