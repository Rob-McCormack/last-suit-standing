#!/usr/bin/env python3
"""Training-scale experiment (pre-registered in scaling.md).

Does more 6-card training make play at other card counts better or worse?
Models: MLP and gradient-boosted trees, 3 seeds each, trained on the first
250 / 500 / 1,500 / 4,000 six-card deals of one fixed stream (prefixes, so
smaller sets are subsets of larger ones). Tested on 4-8 cards against
perfect play, with the same test deals as size_shift.py.

Usage: python3 experiments/scaling.py [--sizes 250,500,1500,4000] [--seeds 0,1,2] [--test-deals 200]
"""
import argparse
import os
import random
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import size_shift as Z  # noqa: E402
from size_shift import S, SIZES  # noqa: E402
from size_shift_followup import make_model, Policy, feats_plain, test_sets, ci  # noqa: E402

warnings.filterwarnings("ignore")


def stream_dataset(n_max, seed, tkeys):
    """Positions from the first n_max six-card deals, plus each deal's end index.
    Uses the same deal stream and rollout stream as size_shift.py, so the first
    1,500 deals reproduce that experiment's training data exactly."""
    rng_d = random.Random(100 * seed + 6)
    deals = [d for d in (rng_d.sample(S.DECK, 6) for _ in range(n_max)) if tuple(map(tuple, d)) not in tkeys]
    rng = random.Random(seed * 7 + 6)
    X, y, ends = [], [], []
    for d in deals:
        seen = set()
        for eps in (0.0, 0.25, 0.5, 1.0, 1.0):
            for st in Z.rollout(d, eps, rng):
                if st in seen:
                    continue
                seen.add(st)
                v = Z.value(*st)
                for p in [(0, 1, 2, 3)] + [tuple(rng.sample(range(4), 4)) for _ in range(2)]:
                    X.append(feats_plain(*st, perm=p))
                    y.append(v)
        Z.clear()
        ends.append(len(y))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), ends


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="250,500,1500,4000")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--test-deals", type=int, default=200)
    args = ap.parse_args()
    ns = [int(x) for x in args.sizes.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    t0 = time.time()

    tests = test_sets(args.test_deals)
    tkeys = {tuple(map(tuple, d)) for ds in tests.values() for d in ds}
    greedy = {te: Z.evaluate(Z.Greedy(), tests[te])[:, :2].mean(axis=1) for te in SIZES}

    res = {}  # (kind, n, test size) -> list over seeds of per-deal points lost
    for seed in seeds:
        X, y, ends = stream_dataset(max(ns), seed, tkeys)
        print(f"seed {seed}: {len(ends)} deals, {len(y)} positions [{time.time() - t0:.0f}s]", flush=True)
        for n in ns:
            k = ends[min(n, len(ends)) - 1]
            for kind in ("mlp", "trees"):
                pol = Policy(make_model(kind, seed).fit(X[:k], y[:k]), feats_plain)
                for te in SIZES:
                    res.setdefault((kind, n, te), []).append(Z.evaluate(pol, tests[te])[:, :2].mean(axis=1))
                print(f"  seed {seed} {kind} n={n} ({k} positions) [{time.time() - t0:.0f}s]", flush=True)

    mean = {key: np.mean(v, axis=0) for key, v in res.items()}  # per-deal, averaged over seeds
    rng = np.random.default_rng(0)

    for kind in ("mlp", "trees"):
        print(f"\n{kind}: points lost per game vs perfect play (mean of seeds {seeds})")
        print(f"{'deals':<10}" + "".join(f"{te:>9} cards" for te in SIZES))
        for n in ns:
            print(f"{n:<10}" + "".join(f"{mean[(kind, n, te)].mean():>15.2f}" for te in SIZES))
        print(f"{'greedy':<10}" + "".join(f"{greedy[te].mean():>15.2f}" for te in SIZES))
        lo_n, hi_n = min(ns), max(ns)
        print(f"  change from {lo_n} to {hi_n} deals (negative = improves), 95% CI:")
        for te in SIZES:
            d = mean[(kind, hi_n, te)] - mean[(kind, lo_n, te)]
            a, b = ci(d, rng)
            cls = "IMPROVES" if b < 0 else "INVERSE SCALING" if a > 0 else "flat"
            per_seed = [np.mean(res[(kind, hi_n, te)][i] - res[(kind, lo_n, te)][i]) for i in range(len(seeds))]
            print(f"    {te} cards: {d.mean():+.2f} [{a:+.2f}, {b:+.2f}] {cls}; per seed "
                  + " ".join(f"{x:+.2f}" for x in per_seed))
    if 1500 in ns and {0, 1} <= set(seeds):
        chk = {te: np.mean([res[("mlp", 1500, te)][seeds.index(s)] for s in (0, 1)], axis=0).mean() for te in (6, 7, 8)}
        print(f"\nsanity check, MLP 1500 deals seeds 0-1 (expect 2.42 / 11.05 / 8.70): "
              f"{chk[6]:.2f} / {chk[7]:.2f} / {chk[8]:.2f}")
    print(f"\ntotal time {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
