#!/usr/bin/env python3
"""Matched Adam updates (pre-registered in matched_steps.md).

The checkpoint run compared training sets at the same pass. A pass over the
larger set is many more Adam updates. This run records the first pass whose
update count reaches each budget.

Quick check:
    python3 experiments/matched_steps.py --sizes 12,20 --seeds 0 --budgets 30,60 --test-deals 2
Full run (writes experiments/matched_steps_results.txt):
    python3 experiments/matched_steps.py
"""
import argparse
import math
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import size_shift as Z  # noqa: E402
from checkpoint import held_out, make_mlp  # noqa: E402
from scaling import stream_dataset  # noqa: E402
from size_shift_followup import Policy, ci, feats_plain, test_sets  # noqa: E402

warnings.filterwarnings("ignore")

TEST_SIZES = (6, 7, 8)
BATCH = 256
# From matched_steps.md. Do not edit these to fit a run.
PREDICTIONS = (
    ("a", "At 10,000 updates, 7-card loss improves from 250 to 4,000 deals", "Yes"),
    ("b", "At 20,000 updates, 7-card loss improves from 1,000 to 4,000 deals", "Yes"),
    ("c", "At 40,000 updates, 7-card loss improves from 1,000 to 4,000 deals", "Yes"),
    ("d", "On 4,000 deals, 7-card loss rises from 10,000 to 40,000 updates", "Yes"),
)


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def updates_per_pass(n_rows):
    return int(math.ceil(n_rows / BATCH))


def first_pass(n_rows, budget):
    step = updates_per_pass(n_rows)
    return int(math.ceil(budget / step)), step


def improves(hi, per_seed):
    return hi < 0 and sum(x < 0 for x in per_seed) >= 3


def rises(lo, per_seed):
    return lo > 0 and sum(x > 0 for x in per_seed) >= 3


def label_change(lo, hi, per_seed):
    if improves(hi, per_seed):
        return "improves"
    if rises(lo, per_seed):
        return "RISES"
    if hi < 0:
        return "CI below 0"
    if lo > 0:
        return "CI above 0"
    return "flat"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="250,1000,4000")
    ap.add_argument("--seeds", default="0,1,2,3")
    ap.add_argument("--budgets", default="10000,20000,40000")
    ap.add_argument("--test-deals", type=int, default=200)
    ap.add_argument("--val-deals", type=int, default=60)
    args = ap.parse_args()
    ns = [int(x) for x in args.sizes.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    budgets = [int(x) for x in args.budgets.split(",")]
    official = (ns == [250, 1000, 4000] and seeds == [0, 1, 2, 3]
                and budgets == [10000, 20000, 40000] and args.test_deals == 200
                and args.val_deals == 60)

    out_f = None
    if official:
        out_f = open(os.path.join(os.path.dirname(__file__), "matched_steps_results.txt"), "w")
        sys.stdout = _Tee(sys.__stdout__, out_f)
    t0 = time.time()
    try:
        _run(ns, seeds, budgets, args, official, t0)
    finally:
        sys.stdout = sys.__stdout__
        if out_f is not None:
            out_f.close()


def _run(ns, seeds, budgets, args, official, t0):
    print(f"{'pre-registered run' if official else 'quick check, results not saved'}: "
          f"sizes {ns}, seeds {seeds}, budgets {budgets}, test {args.test_deals}, "
          f"val deals excluded {args.val_deals}", flush=True)
    tests = {n: test_sets(args.test_deals)[n] for n in TEST_SIZES}
    all_tests = test_sets(args.test_deals)
    val = held_out(args.val_deals, 2000)
    tkeys = {tuple(map(tuple, d)) for ds in all_tests.values() for d in ds}
    tkeys |= {tuple(map(tuple, d)) for ds in val.values() for d in ds}

    # (n, budget, test size) -> list over seeds of per-deal losses
    res = {}
    # (seed, n, budget) -> (pass, updates)
    taken = {}
    for seed in seeds:
        X, y, ends = stream_dataset(max(ns), seed, tkeys)
        print(f"seed {seed}: {len(ends)} six-card deals, {len(y)} positions [{time.time() - t0:.0f}s]",
              flush=True)
        for n in ns:
            k = ends[min(n, len(ends)) - 1]
            passes = {}
            for b in budgets:
                epoch, step = first_pass(k, b)
                passes[b] = epoch
            net = make_mlp(seed)
            want = set(passes.values())
            for epoch in range(1, max(want) + 1):
                net.partial_fit(X[:k], y[:k])
                if epoch not in want:
                    continue
                step = updates_per_pass(k)
                for b, at in passes.items():
                    if at != epoch:
                        continue
                    taken[(seed, n, b)] = (epoch, epoch * step)
                    for te in TEST_SIZES:
                        lost = Z.evaluate(Policy(net, feats_plain), tests[te])[:, :2].mean(axis=1)
                        res.setdefault((n, b, te), []).append(lost)
            bits = " ".join(f"{b}:{taken[(seed, n, b)][0]}p/{taken[(seed, n, b)][1]}u" for b in budgets)
            print(f"  seed {seed} n={n} ({k} rows, {updates_per_pass(k)} updates/pass) {bits} "
                  f"[{time.time() - t0:.0f}s]", flush=True)

    rng = np.random.default_rng(0)

    def contrast(rows_hi, rows_lo):
        per = [float(np.mean(h - lo)) for h, lo in zip(rows_hi, rows_lo)]
        diff = np.mean([h - lo for h, lo in zip(rows_hi, rows_lo)], axis=0)
        a, b = ci(diff, rng)
        return diff, a, b, per

    print("\nPoints lost per game (mean of seeds)")
    print(f"{'budget':<10}{'deals':<10}{'updates':<10}" + "".join(f"{te:>9} cards" for te in TEST_SIZES))
    for b in budgets:
        for n in ns:
            upd = int(np.mean([taken[(s, n, b)][1] for s in seeds]))
            vals = [np.mean(res[(n, b, te)], axis=0).mean() for te in TEST_SIZES]
            print(f"{b:<10}{n:<10}{upd:<10}" + "".join(f"{v:>15.2f}" for v in vals))

    # Comparisons named by the predictions. (hi_n, lo_n, budget, test size)
    named = {
        "a": (max(ns) if official else ns[-1], ns[0], budgets[0], 7),
        "b": (ns[-1], ns[len(ns) // 2] if not official else 1000, budgets[1] if len(budgets) > 1 else budgets[-1], 7),
        "c": (ns[-1], ns[len(ns) // 2] if not official else 1000, budgets[2] if len(budgets) > 2 else budgets[-1], 7),
    }
    # Official comparisons are fixed. Quick checks still print the two ends.
    if official:
        pairs = [
            ("a", 4000, 250, 10000, 7),
            ("b", 4000, 1000, 20000, 7),
            ("c", 4000, 1000, 40000, 7),
        ]
    else:
        pairs = [(pid, hi, lo, b, te) for pid, (hi, lo, b, te) in named.items()
                 if hi in ns and lo in ns and hi != lo and b in budgets]

    cmp = {}
    print("\nPaired change (negative = improves), 95% CI; per seed:")
    for pid, hi, lo, b, te in pairs:
        diff, a, c, per = contrast(res[(hi, b, te)], res[(lo, b, te)])
        cmp[pid] = (diff, a, c, per)
        print(f"  {pid}  {te} cards, budget {b}, {hi} minus {lo}: {diff.mean():+.2f} "
              f"[{a:+.2f}, {c:+.2f}] {label_change(a, c, per)}; per seed "
              + " ".join(f"{x:+.2f}" for x in per))

    if official:
        diff, a, c, per = contrast(res[(4000, 40000, 7)], res[(4000, 10000, 7)])
        cmp["d"] = (diff, a, c, per)
        print(f"  d  7 cards, 4,000 deals, 40000 minus 10000 updates: {diff.mean():+.2f} "
              f"[{a:+.2f}, {c:+.2f}] {label_change(a, c, per)}; per seed "
              + " ".join(f"{x:+.2f}" for x in per))
        for te in TEST_SIZES:
            if te == 7:
                continue
            diff, a, c, per = contrast(res[(4000, 10000, te)], res[(250, 10000, te)])
            print(f"  reported {te} cards, budget 10000, 4000 minus 250: {diff.mean():+.2f} "
                  f"[{a:+.2f}, {c:+.2f}] {label_change(a, c, per)}; per seed "
                  + " ".join(f"{x:+.2f}" for x in per))

    print("\nPredictions (call was fixed before the run):")
    if not official:
        print("  scoring withheld: this is not the pre-registered design")
    else:
        observed = {}
        for pid, _, _call in PREDICTIONS:
            _diff, lo, hi, per = cmp[pid]
            if pid == "d":
                observed[pid] = "Yes" if rises(lo, per) else "No"
            else:
                observed[pid] = "Yes" if improves(hi, per) else "No"
        for pid, text, call in PREDICTIONS:
            print(f"  {pid}  observed {observed[pid]:<3}  call {call:<3}  {text}")
    print(f"\ntotal time {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
