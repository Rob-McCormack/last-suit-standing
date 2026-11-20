#!/usr/bin/env python3
"""Slot-free encoding control (pre-registered in slot_free.md).

Part A: 6-only vs mix 6+7, 1,500 deals per size (finding 2).
Part B: 250 / 500 / 1,500 / 4,000 six-card prefixes (finding 1).
The 6-only model is Part B's 1,500-deal model, evaluated once and reused.

Predictions and decision rules live in slot_free.md. This file does not change them.

Quick check:
    python3 experiments/slot_free.py --sizes 16,24 --mix-deals 24 --seeds 0 --test-deals 4
Full run (writes experiments/slot_free_results.txt):
    python3 experiments/slot_free.py
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
from size_shift_followup import dataset, make_model, Policy, test_sets, ci  # noqa: E402

warnings.filterwarnings("ignore")

# From slot_free.md. Observed "Yes" means the 95% CI is entirely above 0.
PREDICTIONS = (
    ("a", "Part A, MLP: mix 6+7 loses more than 6-only on 8 cards", "Yes", "Yes"),
    ("b", "Part A, trees: the same", "Yes", "Yes"),
    ("c", "Part B, MLP, 250 -> 4,000 deals: 7-card loss rises", "Yes", "Yes"),
    ("d", "Part B, MLP: 8-card loss rises", "No", "No"),
    ("e4", "Part B, MLP: 4-card loss rises", "No", "No"),
    ("e5", "Part B, MLP: 5-card loss rises", "Unsure", "Yes"),
)

# scaling_results.txt, 6-card points lost at 250/500/1,500/4,000. Fairness: within 0.5.
OLD6 = {
    "mlp": {250: 3.18, 500: 2.65, 1500: 2.42, 4000: 2.03},
    "trees": {250: 2.77, 500: 2.32, 1500: 1.92, 4000: 1.78},
}


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def feats_slotfree(row, s1, s2, off, perm=(0, 1, 2, 3)):
    """No fixed card slots. Ranks are /10, as in size_shift.features.

    Row: suit counts, suit rank-sums, rank counts, neighbor-pair sums
    (lower rank, higher rank, same-suit, same-rank), left and right ends.
    Stacks: P1 minus P2 of suit rank-sums, rank counts, size/4, pair flag.
    Turn: len(row)/8, and whether P1 offers next.
    """
    def stats(cards):
        counts, sums, ranks = [0.0] * 4, [0.0] * 4, [0.0] * 10
        for r, s in cards:
            i = perm[Z.SUIT_INDEX[s]]
            counts[i] += 1.0
            sums[i] += r / 10.0
            ranks[r - 1] += 1.0
        return counts, sums, ranks

    def pair_flag(cards):
        rs = [r for r, _ in cards]
        return 1.0 if len(rs) != len(set(rs)) else 0.0

    def end(c):
        r, s = c
        oh = [0.0] * 4
        oh[perm[Z.SUIT_INDEX[s]]] = 1.0
        return [r / 10.0] + oh

    counts, sums, ranks = stats(row)
    lo = hi = same_suit = same_rank = 0.0
    for i in range(len(row) - 1):
        r1, s1u = row[i]
        r2, s2u = row[i + 1]
        lo += min(r1, r2) / 10.0
        hi += max(r1, r2) / 10.0
        same_suit += 1.0 if s1u == s2u else 0.0
        same_rank += 1.0 if r1 == r2 else 0.0
    _, a_sums, a_ranks = stats(s1)
    _, b_sums, b_ranks = stats(s2)
    f = counts + sums + ranks + [lo, hi, same_suit, same_rank] + end(row[0]) + end(row[-1])
    f += [x - y for x, y in zip(a_sums, b_sums)]
    f += [x - y for x, y in zip(a_ranks, b_ranks)]
    f += [(len(s1) - len(s2)) / Z.MAX_STACK, pair_flag(s1) - pair_flag(s2)]
    f += [len(row) / Z.MAX_ROW, 1.0 if off == 1 else 0.0]
    return f


def _check_feats():
    row = ((1, "S"), (1, "H"), (10, "S"))
    f = feats_slotfree(row, ((2, "D"), (2, "C")), (), 1, (0, 1, 2, 3))
    exp = ([2, 1, 0, 0] + [1.1, 0.1, 0, 0] + [2, 0, 0, 0, 0, 0, 0, 0, 0, 1]
           + [0.2, 1.1, 0, 1] + [0.1, 1, 0, 0, 0] + [1.0, 1, 0, 0, 0]
           + [0, 0, 0.2, 0.2] + [0, 2, 0, 0, 0, 0, 0, 0, 0, 0] + [0.5, 1] + [3 / 8, 1])
    if len(f) != 50 or any(abs(a - b) > 1e-6 for a, b in zip(f, exp)):
        raise AssertionError(f"slot-free features mismatch: {list(zip(f, exp))}")
    g = feats_slotfree(row, ((2, "D"), (2, "C")), (), 1, (1, 0, 2, 3))
    if g[0] != 1 or g[1] != 2:
        raise AssertionError("suit relabel did not swap S and H")
    h = feats_slotfree(((5, "C"),), (), (), 2)
    if len(h) != 50 or h[3] != 1 or abs(h[7] - 0.5) > 1e-6 or h[12] != 1:
        raise AssertionError("one-card row encoded wrong")
    if any(h[i] != 0 for i in range(18, 22)) or abs(h[48] - 0.125) > 1e-6 or h[49] != 0:
        raise AssertionError("one-card pairs or turn encoded wrong")
    p = feats_slotfree(((3, "D"), (4, "D")), (), (), 1)
    if abs(p[18] - 0.3) > 1e-6 or abs(p[19] - 0.4) > 1e-6 or p[20] != 1 or p[21] != 0:
        raise AssertionError("neighbor pair sums encoded wrong")
    # The unused suit-count return is only there so a future edit cannot drop it silently.
    both = feats_slotfree(((1, "S"),), ((1, "S"),), ((1, "H"),), 1)
    if abs(both[32] - 0.1) > 1e-6 or abs(both[33] + 0.1) > 1e-6 or abs(both[46]) > 1e-6:
        raise AssertionError("stack differences encoded wrong")


def stream_dataset(n_max, seed, tkeys, featfn):
    """Copy of scaling.stream_dataset, with the feature function passed in.

    Same six-card deal stream and rollout stream, so prefixes match that experiment.
    """
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
                    X.append(featfn(*st, perm=p))
                    y.append(v)
        Z.clear()
        ends.append(len(y))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), ends


def seven_dataset(n_train, seed, tkeys, featfn):
    """7-card deals from the same stream as size_shift_followup.train_policy."""
    rng = random.Random(100 * seed + 7)
    deals = [d for d in (rng.sample(S.DECK, 7) for _ in range(n_train))
             if tuple(map(tuple, d)) not in tkeys]
    X, y = dataset(deals, random.Random(seed * 7 + 7), featfn)
    return X, y, len(deals)


def _check_stream():
    from scaling import stream_dataset as orig
    from size_shift_followup import feats_plain
    a = orig(2, 0, set())
    b = stream_dataset(2, 0, set(), feats_plain)
    if a[2] != b[2] or not np.allclose(a[0], b[0]) or not np.allclose(a[1], b[1]):
        raise AssertionError("stream_dataset does not match scaling.py")


def _yn(lo):
    return "Yes" if lo > 0 else "No"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="250,500,1500,4000")
    ap.add_argument("--mix-deals", type=int, default=1500)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--test-deals", type=int, default=200)
    args = ap.parse_args()
    ns = [int(x) for x in args.sizes.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    official = (ns == [250, 500, 1500, 4000] and seeds == [0, 1, 2]
                and args.test_deals == 200 and args.mix_deals == 1500)
    _check_feats()
    _check_stream()

    out_f = None
    if official:
        out_f = open(os.path.join(os.path.dirname(__file__), "slot_free_results.txt"), "w")
        sys.stdout = _Tee(sys.__stdout__, out_f)
    t0 = time.time()
    try:
        _run(ns, seeds, args.mix_deals, args.test_deals, official, t0)
    finally:
        sys.stdout = sys.__stdout__
        if out_f is not None:
            out_f.close()


def _run(ns, seeds, mix_n, n_test, official, t0):
    print("feature self-check passed (50 inputs); stream matches scaling.py on 2 deals")
    print(f"{'pre-registered run' if official else 'quick check, results not saved'}: "
          f"sizes {ns}, mix {mix_n}, seeds {seeds}, test deals {n_test}", flush=True)

    tests = test_sets(n_test)
    tkeys = {tuple(map(tuple, d)) for ds in tests.values() for d in ds}
    greedy = {te: Z.evaluate(Z.Greedy(), tests[te])[:, :2].mean(axis=1) for te in SIZES}

    res = {}    # (kind, n, size) -> per seed, per deal, points lost
    mix = {}    # (kind, size) -> same, for mix 6+7
    train_ns = list(ns)
    if mix_n not in train_ns:
        train_ns.append(mix_n)
    for seed in seeds:
        X, y, ends = stream_dataset(max(train_ns), seed, tkeys, feats_slotfree)
        X7, y7, n7 = seven_dataset(mix_n, seed, tkeys, feats_slotfree)
        print(f"seed {seed}: {len(ends)} six-card deals, {len(y)} positions; "
              f"{n7} seven-card deals, {len(y7)} positions [{time.time() - t0:.0f}s]", flush=True)
        for n in train_ns:
            k = ends[min(n, len(ends)) - 1]
            for kind in ("mlp", "trees"):
                pol = Policy(make_model(kind, seed).fit(X[:k], y[:k]), feats_slotfree)
                for te in SIZES:
                    lost = Z.evaluate(pol, tests[te])[:, :2].mean(axis=1)
                    res.setdefault((kind, n, te), []).append(lost)
                print(f"  seed {seed} {kind} n={n} ({k} positions) [{time.time() - t0:.0f}s]", flush=True)
        k = ends[min(mix_n, len(ends)) - 1]
        for kind in ("mlp", "trees"):
            pol = Policy(make_model(kind, seed).fit(np.concatenate([X[:k], X7]),
                                                     np.concatenate([y[:k], y7])), feats_slotfree)
            for te in SIZES:
                mix.setdefault((kind, te), []).append(Z.evaluate(pol, tests[te])[:, :2].mean(axis=1))
            print(f"  seed {seed} {kind} mix 6+7 ({k}+{len(y7)} positions) [{time.time() - t0:.0f}s]", flush=True)

    mean = {key: np.mean(v, axis=0) for key, v in res.items()}
    mean_m = {key: np.mean(v, axis=0) for key, v in mix.items()}
    rng_b, rng_a = np.random.default_rng(0), np.random.default_rng(0)
    lo_n, hi_n = min(ns), max(ns)
    part_b = {}  # (kind, size) -> (diff, lo, hi)

    print("\nPart B: points lost per game vs perfect play (mean of seeds)")
    for kind in ("mlp", "trees"):
        print(f"\n{kind}")
        print(f"{'deals':<10}" + "".join(f"{te:>9} cards" for te in SIZES))
        for n in ns:
            print(f"{n:<10}" + "".join(f"{mean[(kind, n, te)].mean():>15.2f}" for te in SIZES))
        print(f"{'greedy':<10}" + "".join(f"{greedy[te].mean():>15.2f}" for te in SIZES))
        print(f"  change from {lo_n} to {hi_n} deals (negative = improves), 95% CI:")
        for te in SIZES:
            d = mean[(kind, hi_n, te)] - mean[(kind, lo_n, te)]
            a, b = ci(d, rng_b)
            part_b[(kind, te)] = (d, a, b)
            cls = "IMPROVES" if b < 0 else "INVERSE SCALING" if a > 0 else "flat"
            per = [np.mean(res[(kind, hi_n, te)][i] - res[(kind, lo_n, te)][i]) for i in range(len(seeds))]
            print(f"    {te} cards: {d.mean():+.2f} [{a:+.2f}, {b:+.2f}] {cls}; per seed "
                  + " ".join(f"{x:+.2f}" for x in per))

    print(f"\nPart A: 6-only vs mix 6+7, {mix_n} deals per size")
    print("6-only reuses the Part B model trained on that six-card prefix.")
    print("points lost per game vs perfect play (mean of seeds)")
    print(f"{'':<18}" + "".join(f"{te:>9} cards" for te in SIZES))
    for kind in ("mlp", "trees"):
        for label, src, keyn in (("6-only", mean, mix_n), ("mix 6+7", mean_m, None)):
            vals = [src[(kind, te)].mean() if keyn is None else src[(kind, keyn, te)].mean() for te in SIZES]
            print(f"{kind + ' ' + label:<18}" + "".join(f"{v:>15.2f}" for v in vals))
    print(f"{'greedy':<18}" + "".join(f"{greedy[te].mean():>15.2f}" for te in SIZES))
    print("  mix 6+7 minus 6-only (positive = mix loses more), 95% CI:")
    part_a = {}
    for kind in ("mlp", "trees"):
        for te in SIZES:
            d = mean_m[(kind, te)] - mean[(kind, mix_n, te)]
            a, b = ci(d, rng_a)
            part_a[(kind, te)] = (d, a, b)
            per = [np.mean(mix[(kind, te)][i] - res[(kind, mix_n, te)][i]) for i in range(len(seeds))]
            print(f"    {kind} {te} cards: {d.mean():+.2f} [{a:+.2f}, {b:+.2f}]; per seed "
                  + " ".join(f"{x:+.2f}" for x in per))

    print("\nFairness check: 6-card loss within 0.5 of the old encoding "
          "(MLP 3.18/2.65/2.42/2.03, trees 2.77/2.32/1.92/1.78)")
    judged = official
    for kind in ("mlp", "trees"):
        oks = []
        for n in (250, 500, 1500, 4000):
            if (kind, n, 6) not in mean:
                continue
            got = float(mean[(kind, n, 6)].mean())
            old = OLD6[kind][n]
            ok = abs(got - old) <= 0.5
            oks.append(ok)
            print(f"  {kind} {n}: {got:.2f} vs {old:.2f} (delta {got - old:+.2f}) {'ok' if ok else 'OUT'}")
        if judged and len(oks) == 4:
            print(f"  {kind}: {'PASS' if all(oks) else 'FAIL'}"
                  + ("" if all(oks) else " — uninformative; the encoding lost too much information"))
        else:
            print(f"  {kind}: pass/fail withheld (not the pre-registered design)")

    print("\nPredictions (observed Yes = 95% CI entirely above 0):")
    if not official:
        print("  scoring withheld: this is not the pre-registered design")
    else:
        look = {
            "a": part_a[("mlp", 8)], "b": part_a[("trees", 8)],
            "c": part_b[("mlp", 7)], "d": part_b[("mlp", 8)],
            "e4": part_b[("mlp", 4)], "e5": part_b[("mlp", 5)],
        }
        for pid, text, gemini, mine in PREDICTIONS:
            d, lo, hi = look[pid]
            obs = _yn(lo)
            print(f"  {pid:<2} observed {obs:<3}  Gemini {gemini:<6}  Mine {mine:<6}  "
                  f"{d.mean():+.2f} [{lo:+.2f}, {hi:+.2f}]  {text}")
    print(f"\ntotal time {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
