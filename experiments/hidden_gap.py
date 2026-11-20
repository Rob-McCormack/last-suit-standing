#!/usr/bin/env python3
"""Hidden-gap experiment: equally optimal openings that are not equally robust.

Question
    Under the official rules, some deals have two or more opening offers that are
    exactly tied for best. No test under the official rules can tell them apart.
    How often do those tied openings differ in value under the undo swap rule,
    and by how much?

Definitions (per deal d)
    S_d(a)  exact final margin (P1 - P2) after opening a, perfect play, official rules
    U_d(a)  the same under the undo rule (perfect play re-solved under that rule)
    A_d     the set of openings with the maximum S_d
    G_d     hidden gap = max over A_d of U_d  -  min over A_d of U_d

Pre-registered thresholds (from Kimi's proposal)
    fragile-deal rate (G_d > 0) >= 5%, and mean G_d >= 0.25 points.

Exact: no learning or approximation. Standard library only.
Usage: python3 experiments/hidden_gap.py [--deals 5000] [--seed 2026]
"""
import argparse
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import solver as S  # noqa: E402


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def bootstrap_mean_ci(xs, reps=4000, seed=0):
    rng = random.Random(seed)
    n = len(xs)
    means = sorted(sum(xs[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps))
    return means[int(0.025 * reps)], means[int(0.975 * reps) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deals", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    official, undo = S.Rules(), S.Rules(swap="undo")
    rng = random.Random(args.seed)
    t0 = time.time()

    gaps, tied, regret_left, regret_rand, sizes = [], 0, [], [], []
    example = None
    for _ in range(args.deals):
        d = rng.sample(S.DECK, 6)
        s = S.opening_values(d, official); S.value.cache_clear()
        u = S.opening_values(d, undo); S.value.cache_clear()
        best = max(s)
        A = [i for i, v in enumerate(s) if v == best]
        g = max(u[i] for i in A) - min(u[i] for i in A)
        gaps.append(g)
        tied += len(A) > 1
        sizes.append(len(A))
        regret_left.append(max(u) - u[A[0]])
        regret_rand.append(max(u) - sum(u[i] for i in A) / len(A))
        if example is None or g > example[0]:
            example = (g, d, s, u)

    n = len(gaps)
    fragile = sum(g > 0 for g in gaps)
    lo, hi = wilson(fragile, n)
    mean_gap = sum(gaps) / n
    glo, ghi = bootstrap_mean_ci(gaps)
    frag_gaps = [g for g in gaps if g > 0]

    print(f"{n} random deals (seed {args.seed})")
    print(f"Deals with tied best openings (official rules): {tied / n:.1%}")
    print(f"Fragile deals (hidden gap > 0): {fragile / n:.2%}  (95% Wilson CI {lo:.2%} to {hi:.2%})")
    print(f"Mean hidden gap per deal: {mean_gap:.3f} points  (95% bootstrap CI {glo:.3f} to {ghi:.3f})")
    if frag_gaps:
        print(f"Among fragile deals: mean gap {sum(frag_gaps) / len(frag_gaps):.2f}, max {max(frag_gaps)}")
    print(f"Undo-rule regret of an official-optimal opening: leftmost tie-break "
          f"{sum(regret_left) / n:.2f}, random tie-break {sum(regret_rand) / n:.2f} points/deal")
    ok1, ok2 = lo >= 0.05, glo >= 0.25
    verdict = ("SUPPORTED" if ok1 and ok2 else
               "FALSIFIED" if hi < 0.05 or ghi < 0.25 else "INCONCLUSIVE")
    print(f"Pre-registered prediction (rate >= 5%, mean gap >= 0.25): {verdict}")

    g, d, s, u = example
    names = [f"{S.card_str(d[i])}-{S.card_str(d[i + 1])}" for i in range(5)]
    print(f"\nLargest gap found ({g} points): deal {' '.join(map(S.card_str, d))}")
    print("  opening     official  undo")
    for nm, a, b in zip(names, s, u):
        mark = "  <- tied best (official)" if a == max(s) else ""
        print(f"  {nm:<10}{a:>+9}{b:>+6}{mark}")
    print(f"\ntime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
