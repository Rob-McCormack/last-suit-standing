#!/usr/bin/env python3
"""Ace-suit test: do trained models notice that a low card's suit matters?

Background (idea from ChatGPT): changing the suit of an ace can swing perfect
play more than changing the suit of a 10, because low cards tend to be left
behind, survive to the end and set trump.

Test: matched deals that differ only in the suit of one target card (an ace or
a 10). For each pair, compare the true change in the deal's perfect-play value
with the change in a trained model's estimate of the opening position.

Prediction (stated before running): the models underrate the ace. Their
estimated change, relative to the true change, is smaller for the ace than for
the 10.

Usage: python3 experiments/ace_effect.py [--pairs 400] [--seed 3]
"""
import argparse
import os
import random
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import solver as S  # noqa: E402
from size_shift_followup import train_policy, feats_plain, test_sets, ci  # noqa: E402

warnings.filterwarnings("ignore")


def matched_pairs(n, rng):
    """(deal_a, deal_b, target rank): identical except the target card's suit."""
    out = []
    while len(out) < n:
        rank = rng.choice((1, 10))
        others = rng.sample([c for c in S.DECK if c[0] != rank], 5)
        free = [s for s in S.SUITS if (rank, s) not in others]
        if len(free) < 2:
            continue
        sa, sb = rng.sample(free, 2)
        pos = rng.randrange(6)
        a = others[:pos] + [(rank, sa)] + others[pos:]
        b = others[:pos] + [(rank, sb)] + others[pos:]
        out.append((a, b, rank))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    pairs = matched_pairs(args.pairs, rng)

    tests = test_sets(200)
    tkeys = {tuple(map(tuple, d)) for ds in tests.values() for d in ds}
    tkeys |= {tuple(map(tuple, d)) for a, b, _ in pairs for d in (a, b)}
    cache = {}
    models = {kind: train_policy((6,), 0, kind, feats_plain, 1500, tkeys, cache) for kind in ("mlp", "trees")}

    rows = []  # rank, true change, est change per model, true flip
    for a, b, rank in pairs:
        va, vb = S.solve(a), S.solve(b)
        S.value.cache_clear()
        est = {}
        for kind, pol in models.items():
            x = np.array([feats_plain(tuple(a), (), (), 1), feats_plain(tuple(b), (), (), 1)], dtype=np.float32)
            pa, pb = pol.net.predict(x)
            est[kind] = abs(pa - pb)
        sg = lambda v: (v > 0) - (v < 0)
        rows.append((rank, abs(va - vb), est["mlp"], est["trees"], sg(va) != sg(vb)))

    arr = np.array([(r, t, m, g, f) for r, t, m, g, f in rows], dtype=float)
    rngb = np.random.default_rng(0)
    print(f"{len(rows)} matched pairs (seed {args.seed}); models trained on 6 cards, 1,500 deals, seed 0\n")
    print(f"{'target':<8}{'pairs':>6}{'true change':>13}{'winner flips':>14}{'MLP est.':>10}{'Trees est.':>12}"
          f"{'MLP / true':>12}{'Trees / true':>14}")
    ratio = {}
    for rank, name in ((1, "ace"), (10, "ten")):
        sub = arr[arr[:, 0] == rank]
        t, m, g = sub[:, 1].mean(), sub[:, 2].mean(), sub[:, 3].mean()
        ratio[name] = (m / t, g / t)
        print(f"{name:<8}{len(sub):>6}{t:>13.2f}{sub[:, 4].mean():>14.1%}{m:>10.2f}{g:>12.2f}{m / t:>12.2f}{g / t:>14.2f}")

    ace, ten = arr[arr[:, 0] == 1], arr[arr[:, 0] == 10]
    d_true = ace[:, 1].mean() - ten[:, 1].mean()
    print(f"\ntrue: ace change minus ten change = {d_true:+.2f} points")
    for k, label in ((2, "MLP"), (3, "Trees")):
        d = ace[:, k].mean() - ten[:, k].mean()
        print(f"{label}: ace change minus ten change = {d:+.2f} points "
              f"(captures {d / d_true:.0%} of the true ace-over-ten gap)" if d_true else "")
    for k, label in ((2, "MLP"), (3, "Trees")):
        # bootstrap the ratio difference (ace ratio - ten ratio)
        diffs = []
        for _ in range(4000):
            ia = rngb.integers(0, len(ace), len(ace)); it = rngb.integers(0, len(ten), len(ten))
            diffs.append(ace[ia, k].mean() / ace[ia, 1].mean() - ten[it, k].mean() / ten[it, 1].mean())
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        r_a, r_t = ratio["ace"][k - 2], ratio["ten"][k - 2]
        verdict = "CONFIRMED (underrates the ace)" if hi < 0 else "not confirmed" + (
            " (overrates the ace)" if lo > 0 else " (no clear difference)")
        print(f"Prediction, {label}: est/true ratio ace {r_a:.2f} vs ten {r_t:.2f}; "
              f"difference {r_a - r_t:+.2f} [{lo:+.2f}, {hi:+.2f}] -> {verdict}")


if __name__ == "__main__":
    main()
